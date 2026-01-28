#include "autograd/Engine.h"
#include "autograd/Functions.h"
#include "core/AutogradMeta.h"
#include "core/TensorImpl.h"
#include "ops/TensorOps.h"
#include "utils/ThreadPool.h"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <stdexcept>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace OwnTensor {
namespace autograd {

// =============================================================================
// Gradient Vector Pool
// =============================================================================
class GradientVectorPool {
    std::vector<std::vector<Tensor>> pool_;
    std::mutex mutex_;
    
public:
    static GradientVectorPool& instance() {
        static GradientVectorPool pool;
        return pool;
    }

    std::vector<Tensor> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            std::vector<Tensor> vec = std::move(pool_.back());
            pool_.pop_back();
            return vec;
        }
        std::vector<Tensor> vec;
        vec.reserve(4);
        return vec;
    }

    void release(std::vector<Tensor>&& vec) {
        vec.clear();
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(vec));
    }
};

// =============================================================================
// Topological Sort (Optimized)
// =============================================================================
std::vector<std::shared_ptr<Node>> topological_sort(const Tensor& root) {
    std::vector<std::shared_ptr<Node>> result;
    std::unordered_set<Node*> visited;
    auto root_fn = root.grad_fn();
    if (!root_fn) return result;
    
    // Iterative Post-Order DFS
    std::vector<std::shared_ptr<Node>> stack;
    
    struct Frame {
        std::shared_ptr<Node> node;
        size_t next_edge_idx = 0;
    };
    
    std::vector<Frame> call_stack;
    if (root_fn) {
        call_stack.push_back({root_fn, 0});
        visited.insert(root_fn.get());
    }
    
    while (!call_stack.empty()) {
        auto& frame = call_stack.back();
        auto node = frame.node;
        
        bool pushed_child = false;
        const auto& edges = node->next_edges();
        
        while (frame.next_edge_idx < edges.size()) {
            const auto& edge = edges[frame.next_edge_idx];
            frame.next_edge_idx++;
            
            if (edge.is_valid()) {
                if (visited.find(edge.function.get()) == visited.end()) {
                    visited.insert(edge.function.get());
                    call_stack.push_back({edge.function, 0});
                    pushed_child = true;
                    break;
                }
            }
        }
        
        if (!pushed_child) {
            result.push_back(node);
            call_stack.pop_back();
        }
    }
    
    std::reverse(result.begin(), result.end());
    return result;
}

// =============================================================================
// Backward Engine
// =============================================================================

// Execution State for a single node
struct NodeTask {
    std::mutex mutex;
    std::unordered_map<uint32_t, std::vector<Tensor>> input_grads_map;
    int dependencies = 0;
    bool scheduled = false;
    uint32_t max_output_nr = 0;

    NodeTask() = default;
};

// Shared Context for the entire Backward Pass
struct BackwardContext {
    std::mutex state_mutex;
    std::condition_variable cv;
    std::atomic<int> active_tasks{0};
    
    // Map Node* -> NodeTask
    // Access to this map is read-only during execution
    std::unordered_map<Node*, std::unique_ptr<NodeTask>> graph_tasks;
    
    // Thread Pool (Static)
    // We refer to utils::ThreadPool
};

void backward(const Tensor& root, const Tensor* grad_output) {
    if (!root.requires_grad()) {
        throw std::runtime_error("backward: tensor does not require gradients");
    }

    // 1. Initialize Root Gradient
    Tensor root_grad;
    bool is_scalar = root.ndim() == 0 || root.numel() == 1;
    if (grad_output) {
        root_grad = *grad_output;
    } else if (is_scalar) {
        root_grad = Tensor::ones(root.shape(), TensorOptions()
            .with_dtype(root.dtype())
            .with_device(root.device()));
    } else {
        throw std::runtime_error("backward: non-scalar requires grad_output");
    }

    auto root_fn = root.grad_fn();
    
    // Handle Leaf Root case
    if (!root_fn) {
         if (root.unsafeGetTensorImpl()->has_autograd_meta()) {
            auto* meta = static_cast<AutogradMeta*>(
                root.unsafeGetTensorImpl()->autograd_meta());
            if (meta->has_grad()) {
                Tensor& existing_grad = meta->mutable_grad(root.unsafeGetTensorImpl());
                Tensor new_grad = operator+(existing_grad, root_grad);
                meta->set_grad(new_grad);
            } else {
                meta->set_grad(root_grad);
            }
        }
        return;
    }

    // 2. Discover Graph and Helper Setup
    auto ctx = std::make_shared<BackwardContext>();
    std::vector<Node*> bfs_queue;
    
    bfs_queue.push_back(root_fn.get());
    ctx->graph_tasks[root_fn.get()] = std::make_unique<NodeTask>();
    
    size_t head = 0;
    while(head < bfs_queue.size()) {
        Node* node = bfs_queue[head++];
        
        for (const auto& edge : node->next_edges()) {
            if (edge.is_valid()) {
                Node* next_node = edge.function.get();
                
                auto& task = ctx->graph_tasks[next_node];
                if (!task) {
                    task = std::make_unique<NodeTask>();
                    bfs_queue.push_back(next_node);
                }
                task->dependencies++;
                if (edge.input_nr > task->max_output_nr) {
                    task->max_output_nr = edge.input_nr;
                }
            }
        }
    }

    // 3. Execution Setup
    static utils::ThreadPool thread_pool(std::thread::hardware_concurrency());
    
    // Seed root gradient
    {
        auto& task = ctx->graph_tasks[root_fn.get()];
        uint32_t slot = root.output_nr();
        task->input_grads_map[slot].push_back(root_grad);
        if (slot > task->max_output_nr) task->max_output_nr = slot;
    }
    
    
    // IMPLEMENTATION OF SAFE RECURSIVE TASK
    struct TaskFunctor {
        std::shared_ptr<BackwardContext> ctx;
        Node* node;
        
        void operator()() {
            NodeTask* task = ctx->graph_tasks.at(node).get();
            
            variable_list node_inputs;
            bool has_grad = false;
            {
                std::lock_guard<std::mutex> lock(task->mutex);
                node_inputs.resize(task->max_output_nr + 1);
                for (auto& [slot, grads] : task->input_grads_map) {
                    if (!grads.empty()) {
                        Tensor sum = grads[0];
                        for (size_t i = 1; i < grads.size(); ++i) {
                            sum = operator+(sum, grads[i]);
                        }
                        node_inputs[slot] = sum;
                        has_grad = true;
                    }
                }
                task->input_grads_map.clear();
            }
            
            if (has_grad) {
                variable_list input_grads = (*node)(std::move(node_inputs));
                node->release_saved_variables();
                
                const auto& edges = node->next_edges();
                for (size_t i = 0; i < edges.size() && i < input_grads.size(); ++i) {
                    if (!edges[i].is_valid()) continue;
                    
                    Node* next_node = edges[i].function.get();
                    uint32_t slot = edges[i].input_nr;
                    Tensor& out_grad = input_grads[i];
                    
                    NodeTask* next_task = ctx->graph_tasks.at(next_node).get();
                    bool ready = false;
                    {
                        std::lock_guard<std::mutex> lock(next_task->mutex);
                        if (out_grad.is_valid()) {
                           next_task->input_grads_map[slot].push_back(std::move(out_grad));
                           if (slot > next_task->max_output_nr) next_task->max_output_nr = slot;
                        }
                        next_task->dependencies--;
                        if (next_task->dependencies == 0 && !next_task->scheduled) {
                            next_task->scheduled = true;
                            ready = true;
                        }
                    }
                    
                    if (ready) {
                        schedule(ctx, next_node);
                    }
                }
            }
            
            
            // Atomic decrement
            int remaining = --ctx->active_tasks;
            if (remaining == 0) {
                 std::lock_guard<std::mutex> lk(ctx->state_mutex);
                 ctx->cv.notify_all();
            }
        }
        
        static void schedule(std::shared_ptr<BackwardContext> ctx, Node* node) {
            ctx->active_tasks++;
            // Enqueue a fresh TaskFunctor
            static utils::ThreadPool& pool = get_pool();
            pool.enqueue(TaskFunctor{ctx, node});
        }
        
        static utils::ThreadPool& get_pool() {
             static utils::ThreadPool pool(std::thread::hardware_concurrency());
             return pool;
        }
    };
    
    // Kickoff
    TaskFunctor::schedule(ctx, root_fn.get());
    
    // Wait
    {
        std::unique_lock<std::mutex> lk(ctx->state_mutex);
        ctx->cv.wait(lk, [&] { return ctx->active_tasks.load() == 0; });
    }
}

} // namespace autograd
} // namespace OwnTensor
#include <iostream>
#include "TensorLib.h"
#include "dataloader.cpp"
// #include <mpi.h>
#include <iomanip>
#include <fstream>
#include "Checkpointing/GradMode.h" // For NoGradGuard
#include "Checkpointing/Checkpoint.h"
// #include <format>
#include <chrono>
// #include "include/Error_logs.hpp"
// #include <nvToolsExt.h>
const OwnTensor::Device dev = OwnTensor::device::cuda_available() ? OwnTensor::Device::CUDA : OwnTensor::Device::CPU;
int rank;
bool print = false;
struct Config{
public:
    int64_t batch = 8;
    int64_t val_steps = 250;
    int64_t tokens = 1024;
    int64_t n_embd = 384;
    int64_t V = 50304;
    int64_t n_layers = 20; // Increased layers to really test checkpointing
    int64_t global_batch = 65536;
    int64_t max_steps = 3250;
    float max_lr = 0.5e-4;
    float min_lr = 0.5e-5;
    int64_t warmup_steps = 325;
    Config() = default;
    Config(int64_t batch, int64_t tokens, int64_t n_embd)
        :batch(batch), tokens(tokens), n_embd(n_embd){}
};
class MLP : public OwnTensor::nn::Module{
public:
    MLP(int64_t V, int64_t emb_dim): V(V){
        model = new OwnTensor::nn::Sequential({
            new OwnTensor::nn::Linear(emb_dim, emb_dim * 4),
            new OwnTensor::nn::GeLU(),
            new OwnTensor::nn::Linear(emb_dim * 4, emb_dim)
        });

        register_module(model);  
    }

    MLP(Config config): MLP(config.V, config.n_embd) {}
    

    OwnTensor::Tensor forward(const OwnTensor::Tensor& input) {
        // auto tokemb = lookup_table(input); //the actual B,T,C will happen here.
        auto predicted = model->forward(input); //the predicted value
        
        return predicted; 
        
    }

    
private: 
    OwnTensor::nn::Module* model;
    OwnTensor::nn::Embedding lookup_table;
    OwnTensor::nn::Linear logit_linear;
    int64_t V;
    // std::vector<OwnTensor::nn::Module> model;
};

class LayerBlock : public OwnTensor::nn::Module {
public:
    LayerBlock(Config config) {
        mlp = new MLP(config);
        ln = new OwnTensor::nn::LayerNorm(config.n_embd);
        register_module(mlp);
        register_module(ln);
    }

    OwnTensor::Tensor forward(const OwnTensor::Tensor& input) override {
        return ln->forward(mlp->forward(input));
    }

private:
    MLP* mlp;
    OwnTensor::nn::LayerNorm* ln;
};

class Model : public OwnTensor::nn::Module {
public:
    Model(Config config): config(config){
        lookup_table = OwnTensor::nn::Embedding(config.V, config.n_embd);
        pos_emb = OwnTensor::nn::Embedding(config.tokens, config.n_embd);
        register_module(lookup_table);
        register_module(pos_emb);

        std::vector<Module*> blocks;
        for(int i = 0; i < config.n_layers; i++) {
            blocks.push_back(new LayerBlock(config));
        }
        
        // Wrap blocks in a Sequential model for checkpointing
        transformer_blocks = std::make_shared<OwnTensor::nn::Sequential>(blocks);
        register_module(transformer_blocks.get());

        logit_linear = OwnTensor::nn::Linear(config.n_embd, config.V, false);
        register_module(logit_linear);

        ln_f = OwnTensor::nn::LayerNorm(config.n_embd);
        register_module(ln_f);

        // pos indices
        pos = OwnTensor::Tensor({{1, config.tokens}}, OwnTensor::Dtype::UInt16);
        uint16_t* pos_ptr = pos.data<uint16_t>();
        for(int i = 0; i < config.tokens; i++){
            pos_ptr[i] = i;
        }
        pos.to(OwnTensor::DeviceIndex(dev, rank));  
    }

    OwnTensor::Tensor forward(const OwnTensor::Tensor& input){
        auto tokemb = lookup_table(input);
        auto posemb = pos_emb(pos);
        auto x = tokemb + posemb;

        // Use checkpoint_sequential to save memory
        // Split 20 layers into 5 segments (4 layers each)
        OwnTensor::variable_list outputs = OwnTensor::autograd::checkpoint_sequential(
            transformer_blocks, 5, {x});
        
        x = outputs[0];
        x = ln_f.forward(x);
        return x;
    }

    OwnTensor::Tensor calc_loss(OwnTensor::Tensor& predicted, OwnTensor::Tensor& target){
        auto B = predicted.shape().dims[0];
        auto T = predicted.shape().dims[1];
        auto logits = logit_linear(predicted);
        logits = logits.reshape({{B*T, config.V}});
        target = target.reshape({{B*T}});
        auto loss = OwnTensor::autograd::sparse_cross_entropy_loss(logits, target);
        return loss;
    }

private:
    std::shared_ptr<OwnTensor::nn::Sequential> transformer_blocks;
    OwnTensor::nn::Embedding lookup_table;
    OwnTensor::nn::Linear logit_linear;
    OwnTensor::nn::Embedding pos_emb;
    OwnTensor::nn::LayerNorm ln_f;
    OwnTensor::Tensor pos;
    Config config;
};

float get_lr(int step, float max_lr, float min_lr, int warmup_steps, int max_steps) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step > max_steps) {
        return min_lr;
    }
    float decay_ratio = static_cast<float>(step - warmup_steps) / static_cast<float>(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}

OwnTensor::Tensor validation(Model& model, DataLoaderLite& dl_val){
    int val_loss_steps = 20;
    dl_val.reset();
    Config config;
    OwnTensor::autograd::NoGradGuard guard; // Disable gradient tracking for validation
    auto val_loss_accum = OwnTensor::Tensor::zeros({{1}}, model.parameters()[0].opts());
    for(auto val_step = 0; val_step < val_loss_steps; val_step++){
        auto batch = dl_val.next_batch();
        auto input = batch.input;
        auto target = batch.target;

        auto prediction = model(input);

        auto val_loss = model.calc_loss(prediction, target);
        val_loss /= val_loss_steps;
        val_loss_accum += val_loss;
    }

    return val_loss_accum;

}

int main(){
    int world_size;
    int64_t steps = 20;
    rank = 0;
    std::ofstream log_file("/home/blubridge-029/agtensor/Tensor-Implementations/Tests/training/logs/log1.csv");
    log_file << "step,loss,val_loss,lr,tok_per_s\n";
    log_file << std::fixed << std::setprecision(6); 
    log_file.flush();
    Config config;
    int grad_accum_steps = config.global_batch / (config.batch * config.tokens);
    DataLoaderLite dl_train(config.batch, config.tokens, 0, 1, "train" ,"/home/blubridge-029/agtensor/Tensor-Implementations/Tests/training/data", 100000000);
    DataLoaderLite dl_val(config.batch, config.tokens, 0, 1, "val" ,"/home/blubridge-029/agtensor/Tensor-Implementations/Tests/training/data", 100000000);
    dl_train.reset();
    Model model(config);
    int64_t total_parameters = 0;
    for(auto param: model.parameters()){
        total_parameters += param.numel();
    }   
    std::cout << "Total parameters: " << total_parameters << std::endl;
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));
    std::vector<OwnTensor::Tensor*> model_param;
    // for(auto& param : model.parameters()){
    //     model_param.push_back(&param);
    // }
    auto params = model.parameters();
    OwnTensor::nn::AdamW optim(params, config.max_lr, 0.9f, 0.95f, 1e-8, 0.1f);
    for(auto step = 0; step < config.max_steps; step++){
        auto start = std::chrono::steady_clock::now(); //start_time
        //validation
        OwnTensor::Tensor val_loss_accum = OwnTensor::Tensor::zeros({{1}}, model.parameters()[0].opts());
        float val_loss_float = 0.0f;
        if(step % config.val_steps == 0 || step == config.max_steps - 1){
            val_loss_accum = validation(model, dl_val);
            val_loss_float = val_loss_accum.to_cpu().data<float>()[0];
            std::cout << "Validation loss at step " << step << ": " << val_loss_float << std::endl; 
        }
        //training
        model.zero_grad();
        OwnTensor::TensorOptions opts;
        opts.device = OwnTensor::DeviceIndex(dev, rank);
        auto loss_acc = OwnTensor::Tensor::zeros({{1}}, opts);
        std::string temp = "Step: " + std::to_string(step);
        // nvtxRangePush(temp.c_str());
        for(auto micro_step = 0; micro_step < grad_accum_steps; micro_step++){
            auto batch = dl_train.next_batch();
            auto input = batch.input;
            auto target = batch.target;
            // nvtxRangePush("Forward Start");
            auto prediction = model(input);
            // nvtxRangePop();
            auto loss = model.calc_loss(prediction, target);
            loss = loss/grad_accum_steps;
            loss_acc += loss;
            // nvtxRangePush("Backend Start");
            loss.backward();
            // nvtxRangePop();
        }
        // nvtxRangePop();
        auto loss_acc_float = loss_acc.to_cpu().data<float>()[0];
        float lr_ = get_lr(step, config.max_lr,  config.min_lr, config.warmup_steps, config.max_steps);
        optim.set_lr(lr_);
        // std::cout << "===================================================\n";
        // std::cout << "STEP: " << step << std::endl;
        // std::cout << "===================================================\n";
        // for(auto param : model.parameters()){
        //     param.grad_view().display();
        // }
        optim.step();

        auto stop = std::chrono::steady_clock::now(); //stop_time
        auto time_taken = std::chrono::duration<float>(stop - start);
        float throughput = (float)config.global_batch / time_taken.count();
        if (!log_file.is_open()) {
            std::cerr << "Failed to open loss_log.csv for writing!" << std::endl;
        }
        log_file << step << 
                    "," << loss_acc_float <<
                    "," << val_loss_float <<
                    "," << lr_ << 
                    "," << throughput << std::endl;
        std::cout << "Step: " << step << ", " << "Loss: " << loss_acc_float << ", "  << "Time Taken: " << time_taken.count() << "sec" << ", " << "Throughput: " << throughput << "tok/sec." << std::endl;
    }
    log_file.close();
    // MPI_Finalize();
    return 0;
}

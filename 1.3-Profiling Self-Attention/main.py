import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as profiler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  
        self.heads = heads  
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by number of heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0] 
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.view(N, -1, self.heads, self.head_dim)
        keys = keys.view(N, -1, self.heads, self.head_dim)
        queries = queries.view(N, -1, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, -1, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

def profile_self_attention(input_lengths, embed_size=512, heads=1, device='cuda', warmup_iters=5, num_trials=10):
    flops_list = []
    memory_list = []
    time_list = []

    flops_se_list = []
    memory_se_list = []
    time_se_list = []

    self_attention = SelfAttention(embed_size, heads).to(device)

    for length in input_lengths:
        flops_trials = []
        memory_trials = []
        time_trials = []
        
        for _ in range(num_trials):
            values = torch.randn((length, embed_size)).to(device)
            keys = torch.randn((length, embed_size)).to(device)
            queries = torch.randn((length, embed_size)).to(device)

            for _ in range(warmup_iters):
                _ = self_attention(values, keys, queries)
            
            torch.cuda.empty_cache()

            with profiler.profile(record_shapes=True, profile_memory=True) as prof:
                with torch.profiler.record_function("self_attention"):
                    _ = self_attention(values, keys, queries)
            
            flops = sum([event.cpu_time for event in prof.key_averages()])
            memory = torch.cuda.max_memory_allocated(device)
            start_time = time.time()
            _ = self_attention(values, keys, queries)
            end_time = time.time()
            time_taken = end_time - start_time
            
            flops_trials.append(flops)
            memory_trials.append(memory)
            time_trials.append(time_taken)

            torch.cuda.reset_max_memory_allocated(device)
        
        flops_mean = np.mean(flops_trials)
        memory_mean = np.mean(memory_trials)
        time_mean = np.mean(time_trials)

        flops_se = np.std(flops_trials) / np.sqrt(num_trials)
        memory_se = np.std(memory_trials) / np.sqrt(num_trials)
        time_se = np.std(time_trials) / np.sqrt(num_trials)

        flops_list.append(flops_mean)
        memory_list.append(memory_mean)
        time_list.append(time_mean)

        flops_se_list.append(flops_se)
        memory_se_list.append(memory_se)
        time_se_list.append(time_se)

    return flops_list, memory_list, time_list, flops_se_list, memory_se_list, time_se_list

def plot_results(input_lengths, flops_list, memory_list, time_list, flops_se_list, memory_se_list, time_se_list):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot FLOPS with error bars
    axs[0].errorbar(input_lengths, flops_list, yerr=flops_se_list, fmt='-o', capsize=5)
    axs[0].set_title("FLOPS vs Input Length")
    axs[0].set_xlabel("Input Length")
    axs[0].set_ylabel("FLOPS")

    # Plot Memory Usage with error bars
    axs[1].errorbar(input_lengths, memory_list, yerr=memory_se_list, fmt='-o', capsize=5)
    axs[1].set_title("Memory Usage vs Input Length")
    axs[1].set_xlabel("Input Length")
    axs[1].set_ylabel("Memory Usage (Bytes)")

    # Plot Time Taken with error bars
    axs[2].errorbar(input_lengths, time_list, yerr=time_se_list, fmt='-o', capsize=5)
    axs[2].set_title("Time Taken vs Input Length")
    axs[2].set_xlabel("Input Length")
    axs[2].set_ylabel("Time Taken (Seconds)")

    plt.tight_layout()
    plt.show()
    plt.savefig('main_plot_error.png', dpi=300)

# Set parameters
input_lengths = [10, 100, 1000, 10000, 100000]
flops_list, memory_list, time_list, flops_se_list, memory_se_list, time_se_list = profile_self_attention(input_lengths)

# Plot results with error bars
plot_results(input_lengths, flops_list, memory_list, time_list, flops_se_list, memory_se_list, time_se_list)


# -*- coding: utf-8 -*-

def top_k_subarray_sums(nums, k, top_n=3):
    L = len(nums)

    if L == 0 or k <= 0 or k > L:
        raise ValueError("ERROR")

    results = []

    window_sum = sum(nums[:k])
    results.append((window_sum, 0))

    for i in range(k, L):
        window_sum += nums[i]
        window_sum -= nums[i - k]
        start_idx = i - k + 1
        results.append((window_sum, start_idx))

    results.sort(key=lambda x: x[0], reverse=True)

    top_results = results[:top_n]

    final_results = []
    for s, start in top_results:
        indices = list(range(start, start + k))
        final_results.append((s, indices))

    return final_results

# INPUT PROBE RESULTS
# ==========================
# IDN
# ==========================

IDN_BERT_base_cased_LIN_1k_ORIG = [0.45, 0.47, 0.475, 0.465, 0.46, 0.44, 0.445, 0.47, 0.47, 0.465, 0.44, 0.39, 0.405]
IDN_StarEncoder_LIN_1k_ORIG = [0.415, 0.4, 0.41, 0.46, 0.525, 0.51, 0.445, 0.49, 0.48, 0.495, 0.475, 0.51, 0.475]
IDN_CodeBERT_base_LIN_1k_ORIG = [0.44, 0.46, 0.45, 0.475, 0.45, 0.505, 0.505, 0.47, 0.49, 0.455, 0.485, 0.505, 0.42]
IDN_CodeGPT_small_java_LIN_1k_ORIG = [0.46, 0.38, 0.425, 0.42, 0.46, 0.44, 0.45, 0.445, 0.5, 0.455, 0.425, 0.46, 0.405]
IDN_PolyCoder_small_LIN_1k_ORIG = [0.455, 0.425, 0.43, 0.405, 0.455, 0.465, 0.45, 0.47, 0.495, 0.51, 0.53, 0.42, 0.51]
IDN_CodeGen_small_multi_LIN_1k_ORIG = [0.475, 0.475, 0.485, 0.48, 0.52, 0.5, 0.49, 0.51, 0.52, 0.495, 0.505, 0.525, 0.51, 0.515, 0.49, 0.44, 0.47, 0.445, 0.47, 0.44, 0.465]

# ==========================
# AST
# ==========================

AST_BERT_base_cased_LIN_1k_ORIG = [0.39, 0.865, 0.825, 0.79, 0.44, 0.765, 0.805, 0.765, 0.625, 0.8, 0.74, 0.815, 0.88]
AST_StarEncoder_LIN_1k_ORIG = [0.165, 0.85, 0.895, 0.885, 0.91, 0.895, 0.91, 0.9, 0.91, 0.91, 0.905, 0.895, 0.88]
AST_CodeBERT_base_LIN_1k_ORIG = [0.205, 0.9, 0.895, 0.875, 0.815, 0.38, 0.285, 0.245, 0.36, 0.81, 0.815, 0.81, 0.84]
AST_CodeGPT_small_java_LIN_1k_ORIG = [0.89, 0.895, 0.9, 0.865, 0.885, 0.88, 0.885, 0.88, 0.88, 0.86, 0.895, 0.885, 0.89]
AST_PolyCoder_small_LIN_1k_ORIG = [0.895, 0.89, 0.88, 0.89, 0.895, 0.86, 0.88, 0.905, 0.865, 0.9, 0.885, 0.905, 0.9]
AST_CodeGen_small_multi_LIN_1k_ORIG = [0.93, 0.94, 0.935, 0.91, 0.91, 0.91, 0.86, 0.865, 0.91, 0.875, 0.92, 0.875, 0.9, 0.9, 0.93, 0.935, 0.935, 0.91, 0.93, 0.9, 0.905]

# ==========================
# CPX
# ==========================

CPX_BERT_base_cased_LIN_1k_ORIG = [0.255, 0.25, 0.255, 0.26, 0.25, 0.29, 0.28, 0.295, 0.29, 0.24, 0.24, 0.265, 0.22]
CPX_StarEncoder_LIN_1k_ORIG = [0.295, 0.275, 0.285, 0.29, 0.325, 0.32, 0.305, 0.3, 0.305, 0.28, 0.295, 0.295, 0.295]
CPX_CodeBERT_base_LIN_1k_ORIG = [0.24, 0.275, 0.28, 0.265, 0.3, 0.295, 0.255, 0.275, 0.28, 0.26, 0.285, 0.275, 0.275]
CPX_CodeGPT_small_java_LIN_1k_ORIG = [0.24, 0.205, 0.225, 0.24, 0.26, 0.26, 0.225, 0.18, 0.165, 0.205, 0.18, 0.19, 0.21]
CPX_PolyCoder_small_LIN_1k_ORIG = [0.21, 0.255, 0.285, 0.315, 0.3, 0.285, 0.27, 0.285, 0.29, 0.31, 0.27, 0.235, 0.24]
CPX_CodeGen_small_multi_LIN_1k_ORIG = [0.175, 0.205, 0.26, 0.26, 0.29, 0.3, 0.275, 0.305, 0.295, 0.28, 0.28, 0.27, 0.29, 0.275, 0.295, 0.275, 0.28, 0.285, 0.265, 0.23, 0.23]


def main():
    k = 2
    top_n = 3

    all_lists = {
        name: value
        for name, value in globals().items()
        if isinstance(value, list) and name.startswith(("IDN_", "AST_", "CPX_"))
    }

    for name, nums in sorted(all_lists.items()):
        try:
            top_results = top_k_subarray_sums(nums, k, top_n)

            print("=" * 60)
            print("列表名:", name)

            for rank, (max_sum, indices) in enumerate(top_results, 1):
                subarray = [nums[i] for i in indices]

                print("Top {}:".format(rank))
                print("  对应索引:", indices)
                print("  最大和:", round(max_sum, 6))
                print("  对应子数组:", subarray)

        except ValueError as e:
            print("=" * 60)
            print("列表名:", name)
            print("错误:", e)


if __name__ == "__main__":
    main()

Treebeard: An Optimizing Compiler for Decision Tree Based ML Inference


## 流程

输入一个决策树数据结构，compiler通过一系列 IR 转化，将决策树数据结构转化为对CPU来说更加友好的数据结构，从而加速决策树上的推理过程。然后，因为自己有关于决策树的domain knowledge，因此在做codegen比如循环展开，循环交换的时候，可以选择到更好的codegen方式。整个项目基于MLIR实现，因此一些简单的优化比如使用 OpenMP 做并行，可以直接用MLIR。




## 图片


TREEBEARD IR lowering 和 optimization 细节

显示了 TREEBEARD IR 中的三个抽象级别。 高级 IR 是基于树的 IR，用于执行模型级优化，中级 IR 用于独立于内存布局的循环优化，低级 IR 允许我们执行向量化和其他与内存布局相关的优化。





Figure 6: Sparse representation with tile size nt = 3. Leaves l4, l5, l6 and l7 are moved into the leaves array. Extra hops are added for l1, l2 and l3 as T2 is a non-leaf tile. The new leaves added as children of l1, l2 and l3 are moved to the leaves array.




Figure 9: Geomean speedup (over all benchmarks) of TREEBEARD over XGBoost and Treelite on single-core over several batch sizes. 



Table I: List of benchmark datasets and their parameters. The column Leaf-biased reports the number of leaf-biased trees per benchmark with 〈α = 0.075,β = 0.9〉 . 


Table II: Space of optimizations explored. 

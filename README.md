# Variational Continual Learning (VCL)
An implementation of the Variational Continual Learning (VCL) algorithms proposed by Nguyen, Li, Bui, and Turner (ICLR 2018).

```
@inproceedings{nguyen2018variational,
  title = {Variational Continual Learning},
  author = {Nguyen, Cuong V. and Li, Yingzhen and Bui, Thang D. and Turner, Richard E.},
  booktitle = {International Conference on Learning Representations},
  year = {2018}
}
```
**To run the Permuted MNIST experiment:**

	python run_permuted.py

**To run the Split MNIST experiment:**

	python run_split.py
	
## Results
### VCL in Deep discriminative models

<p><h4>Permuted MNIST</h4></p>
![](/discriminative/misc/permuted_mnist_main.png)
![](/discriminative/misc/permuted_mnist_coreset_sizes.png)


<h4>Split MNIST</h4>
![](/discriminative/misc/split_mnist_main_part1.png)
![](/discriminative/misc/split_mnist_main_part2.png)
With Variational Generative Replay (VGR):
![](/discriminative/misc/vgr.png)

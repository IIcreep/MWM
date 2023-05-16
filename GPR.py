import torch
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import gpytorch
from gpytorch import kernels, means, models, mlls, settings
from gpytorch import distributions as distr
from gpytorch.kernels import PolynomialKernel

class ExactGPModel(models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = means.ConstantMean()
        # self.covar_module = kernels.ScaleKernel(kernels.RBFKernel())
        # self.covar_module = kernels.ScaleKernel(PolynomialKernel(degree=5, bias=1, power=1))
        self.covar_module = kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=6, ard_num_dims=2))
        # self.covar_module = kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        # rbf_kernel = kernels.ScaleKernel(kernels.RBFKernel())
        # periodic_kernel = kernels.ScaleKernel(kernels.PeriodicKernel())
        # additive_kernel = kernels.AdditiveKernel(rbf_kernel, periodic_kernel)
        # self.covar_module = additive_kernel(train_x,train_y)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distr.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":
    data = pd.read_csv("/Users/hu/Desktop/NNI/data/sigmau.csv")
    features = data.drop("sigmau", axis=1)
    labels = data["sigmau"]
    features = np.array(features)
    labels = np.array(labels)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # 初始化似然和模型
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(features, labels, likelihood)

    # 使用留一法交叉验证来评估模型的性能
    n = len(features) # 训练数据的个数
    rmse = 0 # 均方根误差
    mape = 0
    ave_var = 0
    
    for i in range(n):
        # 将第i个数据点作为测试集，其余作为训练集
        test_x = features[i].view(1, -1)
        # print(test_x)
        test_y = labels[i]
        # print(test_y)
        train_x_cv = torch.cat([features[:i], features[i+1:]])
        # print(train_x_cv)
        train_y_cv = torch.cat([labels[:i], labels[i+1:]])
        # print(train_y_cv)

        # 重新初始化似然和模型
        likelihood_cv = gpytorch.likelihoods.GaussianLikelihood()
        model_cv = ExactGPModel(train_x_cv, train_y_cv, likelihood_cv)

        # 训练模型，使用Adam优化器和最大边缘似然损失函数
        model_cv.train()
        likelihood_cv.train()
        optimizer = torch.optim.Adam(model_cv.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_cv, model_cv)
        num_iter = 80 # 迭代次数
        for j in range(num_iter):
            optimizer.zero_grad()
            output = model_cv(train_x_cv)
            loss = -mll(output, train_y_cv)
            loss.backward()
            optimizer.step()

        # 预测测试点的分布，计算均方根误差
        model_cv.eval()
        likelihood_cv.eval()
        with torch.no_grad(), settings.fast_pred_var():
            pred_dist = likelihood_cv(model_cv(test_x))
            pred_mean = pred_dist.mean.item()
            pred_var = pred_dist.variance.item()
            confidence_down = pred_mean - pred_var**0.5*1.96
            confidence_up = pred_mean + pred_var**0.5*1.96
            ave_var += 2 * pred_var**0.5*1.96
            true_value = test_y.item()
            error1 = abs(true_value - pred_mean)/true_value
            error2 = (pred_mean - true_value) ** 2
            print(error2)
            mape += error1
            rmse += math.sqrt(error2)

    print("RMSE:", rmse/n)
    print("MAPE:", mape/n)
    print("VAR:", ave_var/n)

    # x1 = np.arange(40,510,10)
    # x2 = [0.3]*47
    # my_array = np.column_stack((x1, x2))
    # X = torch.tensor(my_array, dtype=torch.float32)
    # pred_mean = []
    # for item in X:
    #     item = item.unsqueeze(0)
    #     pred_dist = likelihood_cv(model_cv(item))
    #     pred_mean.append(pred_dist.mean.item())
    # print(pred_mean)

    # fig = plt.figure()
    # ax =fig.add_subplot(111, projection="3d")
    # ax.plot_surface(x1,x2,pred_mean,cmap="coolwarm")
    # ax.set_xlabel("X1")
    # ax.set_ylabel("X2")
    # ax.set_zlabel("y")
    # plt.show()

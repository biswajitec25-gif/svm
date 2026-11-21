KERNELS = {
    "title": "SVM Kernels",
    "sections": [
        {"heading": "Linear Kernel", "content": "K(x,y)=x^T y - used for linearly separable data."},
        {"heading": "Polynomial Kernel", "content": "K(x,y)=(gamma*x^T y + r)^d - captures polynomial relationships."},
        {"heading": "RBF Kernel", "content": "K(x,y)=exp(-gamma*||x-y||^2) - handles non-linear data well."},
        {"heading": "Sigmoid Kernel", "content": "K(x,y)=tanh(gamma*x^T y + r) - similar to neural networks."}
    ]
}

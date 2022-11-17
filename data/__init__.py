from .fashion_mnist import get_fashion_mnist

def get_data_loader(
	name='fashion_mnist', batch_size=64, train=True, labels=False
):
    if name=='fashion_mnist':
        return get_fashion_mnist(batch_size, train, labels)
    else:
        raise ValueError(name)
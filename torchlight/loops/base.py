import numpy as np

from torch.autograd import Variable


class BaseLoop():

    def __init__(self):
        self.metrics = []

    ## decorator functions
    def isolate_model_mode(validate):
        def new_fct(fct):
            def wrapper(*args, **kwargs):
                model = args[0].model
                last_mode = model.training
                __ = model.eval() if validate else model.train()
                out = fct(*args, **kwargs)
                __ = model.train() if last_mode else model.eval()
                return out
            return wrapper
        return new_fct

    ## private functions
    def __fcuda(self, tensors):
        return [T.cuda() if self.cuda else T for T in tensors]

    def __update_step(self, inputs, targets):
        self.optimizer.zero_grad()
        loss = self.loss_inference(inputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def __one_batch(self, batch, validate):
        inputs, targets = self.__fcuda(batch)
        if validate:
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)
            loss = self.loss_inference(inputs, targets)
        else:
            inputs = Variable(inputs)
            targets = Variable(targets)
            loss = self.__update_step(inputs, targets)
        return float(loss.data.cpu().numpy()[0])

    def __all_batch(self, dataloader, validate):
        batch_losses = []
        for batch in dataloader:
            loss = self.__one_batch(batch, validate)
            batch_losses.append(loss)
        return np.mean(batch_losses)

    @isolate_model_mode(validate=True)
    def __validate_one_epoch(self, validloader):
        self.reset_metrics()
        valid_loss = self.__all_batch(validloader, validate=True)
        return valid_loss

    @isolate_model_mode(validate=False)
    def __train_one_epoch(self, trainloader):
        self.reset_metrics()
        train_loss = self.__all_batch(trainloader, validate=False)
        return train_loss

    ## public functions
    def loss_inference(self, inputs, targets):
        preds = self.model(inputs)
        for metric in self.metrics:
            metric.append(preds, targets)
        return self.model.loss(preds, targets)

    def register_metric(self, metric):
        self.metrics.append(metric)

    def compute_metrics(self):
        computed_metrics = {}
        for metric in self.metrics:
            computed_metrics[metric.name] = metric.compute()
        return computed_metrics

    def show_metrics(self):
        metric_str = []
        for metric in self.metrics:
            metric_str.append(metric.show())
        return ', '.join(metric_str)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()
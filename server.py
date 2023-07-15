import numpy as np

import models, torch


class Server(object):

    def __init__(self, conf, eval_dataset):

        self.conf = conf
        self.eval_dataset = eval_dataset
        self.global_model = models.get_model(self.conf["model_name"])

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def model_aggregate(self, weight_accumulator):
        device = torch.device("cuda:6")
        for name, data in self.global_model.state_dict().items():
            data = data.to(device)
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64).to(device))
            else:
                data.add_(update_per_layer.to(device))

    def model_eval(self):
        self.global_model.eval()
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        with torch.no_grad():
            for batch_id, batch in enumerate(self.eval_loader):
                data, target = batch
                dataset_size += data.size()[0]
                if torch.cuda.is_available():
                    device = torch.device("cuda:6")
                    data = data.to(device)
                    target = target.to(device)
                    self.global_model = self.global_model.to(device)
                    #data = data.cuda()
                    #target = target.cuda()
                    #self.local_model = self.local_model.cuda()
                output = self.global_model(data)
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]
                # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size

        return acc, total_l

    def label_eval(self):
        self.global_model.eval()
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        correct_list = np.zeros(10)
        total_list = np.zeros(10)

        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.global_model(data)
            pred = output.data.max(1)[1]
            # get the index of the max log-probability

            for i in range(len(target)):
                if pred[i] == target[i]:
                    correct_list[target[i]] += 1
                total_list[target[i]] += 1

        label_acc = np.zeros(10,dtype=float)
        for i in range(len(correct_list)):
            label_acc[i] = correct_list[i] / total_list[i]

        return label_acc

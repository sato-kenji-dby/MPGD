import models, torch



class Client(object):

    def __init__(self, conf, model, train_dataset,id):

        self.conf = conf
        
        self.local_model = models.get_model(self.conf["model_name"])

        self.client_id = id

        self.train_dataset = train_dataset

        self.all_range = list(range(len(self.train_dataset)))


    def local_train(self, model,action):
        device = torch.device("cuda:6")
        #train_indices = self.all_range[sum(action[self.client_id]):sum(self.client_id) + action[self.client_id+1]]
        #print(action)

        
        #print(int(action[self.client_id]))

        #print(int(self.client_id + action[self.client_id]))
        #train_indices = self.all_range[int(action[self.client_id]):int(self.client_id + action[self.client_id])]
        #altered here!!!!!!!!!!!!!!!!!!!!
        train_indices = self.all_range[int(self.client_id*self.conf["self_data"]):int(self.client_id*self.conf["self_data"] + action[self.client_id])]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))
        #print(len(self.train_loader))
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # print(id(model))
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])
        # print(id(self.local_model))
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                #print("bids",batch_id)

                if torch.cuda.is_available():
                    device = torch.device("cuda:6")
                    data = data.to(device)
                    target = target.to(device)
                    self.local_model = self.local_model.to(device)
                    #data = data.cuda()
                    #target = target.cuda()
                    #self.local_model = self.local_model.cuda()
                optimizer.zero_grad()
                #print(data.size())
                
                #print(data.device)
                
                #print(next(model.parameters()).device)
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name].to(device))
            
        # print(diff[name])

        return diff

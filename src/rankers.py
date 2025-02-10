import torch
import torch.nn as nn
import pytorch_lightning as pl

class Ranker1(pl.LightningModule):
    def __init__(self, cbm_buggy, cbm_oracle, lr=1e-9, margin=1.0):
        super(Ranker1, self).__init__()
        self.save_hyperparameters(ignore=["cbm_buggy", "cbm_oracle"])
        #super().__init__()
        self.cbm_buggy = cbm_buggy
        self.cbm_oracle = cbm_oracle
        self.n_concepts = cbm_buggy.n_concepts
        self.n_tasks = cbm_buggy.n_tasks
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.margin = margin


        # Fully connected layers
        img_size = 299*299*3
        weight_top_layer = self.n_concepts*self.n_tasks + self.n_tasks
        inp = img_size + self.n_concepts + weight_top_layer + self.n_tasks
        #print(inp)

        self.lay = torch.nn.Sequential(*[
            torch.nn.Linear(inp, 1),
        ])


    def compute_w(self, model):
        #print("compute_w")
        # Get the last layer (weights w of the model)
        linear_layer = model.c2y_model[0]
        # Flatten weights and biases
        weight_vector = linear_layer.weight.view(-1, 1)
        bias_vector = linear_layer.bias.view(-1, 1)
        # Concatenate into a single vector
        param_vector = torch.cat((weight_vector, bias_vector), dim=0)
        #print(param_vector.size())
        return param_vector

    def forward(self, x, c_logits, w, y_logits):
        """Computes ranking score."""

        #We need to resize, as we have 
        #print(f"Size x: {x.size()}") #[8, ..., ..., ...] # resize to [8, ...]
        #print(f"Size c: {c_logits.size()}") #[8, ...] #no resize needed
        #print(f"Size w: {w.size()}") #[22600, 1] #resize to [8, 22600, 1]
        #print(f"Size y: {y_logits.size()}") #[8, ...] #no resize needed

        # Flatten x and w tensors except the batch dimension
        x_flat = x.view(x.size(0), -1)
        w_flat = w.view(1, -1).repeat(x.size(0), 1)
        
        # Concatenate along the second dimension (feature dimension)
        flattened = torch.cat((x_flat, c_logits, w_flat, y_logits), dim=1)
        
        #print(flattened.size())
        return self.lay(flattened)

    def ranking_loss(self, r_f, r_o):
        target = torch.ones_like(r_f)
        #target = torch.Tensor([1]*batch_size)
        
        ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        output = ranking_loss(r_o, r_f, target)
        return output
        #output.backward()
        #return torch.mean(torch.clamp(self.margin + r_f - r_o, min=0))

    def ranking_loss_GPT(self, r_f, r_o):
        """Pairwise ranking loss (Hinge loss)."""
        return torch.mean(torch.clamp(self.margin + r_f - r_o, min=0))

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        if len(batch) > 3:
            competencies = batch[3]
        else:
            competencies = None
        if len(batch) > 4:
            prev_interventions = batch[4]
        else:
            prev_interventions = None
        return x, y, (c, competencies, prev_interventions)


    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch,batch_idx):
        loss = self.run_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, intervention_idxs=None):
        #Step 1: get the batch unpacked
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #Step 2: for the batch, calculate:
        c_sem, c_logits_f, y_logits_f = self.cbm_buggy(x) #_ = c_sem_f
        c_sem, c_logits_o, y_logits_o = self.cbm_oracle(x) #_ = c_sem_o

        #Step 3: get w_f and w_o
        w_f = self.compute_w(self.cbm_buggy).to(device=self.dev)
        w_o = self.compute_w(self.cbm_oracle).to(device=self.dev)
        
        #Step 4: put both inputs to the self fotward
        r_f = self(x, c_logits_f, w_f, y_logits_f)
        r_o = self(x, c_logits_o, w_o, y_logits_o)

        #Step 5: calculate loss and return
        loss = self.ranking_loss(r_f, r_o)
        return loss

    def run_step(self, batch):
        #Step 1: get the batch unpacked
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #Step 2: for the batch, calculate:
        c_sem, c_logits_f, y_logits_f = self.cbm_buggy(x) #_ = c_sem_f
        c_sem, c_logits_o, y_logits_o = self.cbm_oracle(x) #_ = c_sem_o

        #Step 3: get w_f and w_o
        w_f = self.compute_w(self.cbm_buggy).to(device=self.dev)
        w_o = self.compute_w(self.cbm_oracle).to(device=self.dev)
        
        #Step 4: put both inputs to the self fotward
        r_f = self(x, c_logits_f, w_f, y_logits_f)
        r_o = self(x, c_logits_o, w_o, y_logits_o)

        #Step 5: calculate loss and return
        loss = self.ranking_loss(r_f, r_o)
        return loss
        
    def configure_optimizers(self):
        #print("config optimizers")
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay = 4e-05
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }
    

class Ranker2(pl.LightningModule):
    def __init__(self, cbm_buggy, cbm_oracle, lr=1e-9, margin=1.0):
        super(Ranker2, self).__init__()
        self.save_hyperparameters(ignore=["cbm_buggy", "cbm_oracle"])
        #super().__init__()
        self.cbm_buggy = cbm_buggy
        self.cbm_oracle = cbm_oracle
        self.n_concepts = cbm_buggy.n_concepts
        self.n_tasks = cbm_buggy.n_tasks
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.margin = margin


        # Fully connected layers
        weight_top_layer = self.n_concepts*self.n_tasks + self.n_tasks
        inp = self.n_concepts + weight_top_layer + self.n_tasks
        #print(inp)

        self.lay = torch.nn.Sequential(*[
            torch.nn.Linear(inp, 1),
        ])


    def compute_w(self, model):
        #print("compute_w")
        # Get the last layer (weights w of the model)
        linear_layer = model.c2y_model[0]
        # Flatten weights and biases
        weight_vector = linear_layer.weight.view(-1, 1)
        bias_vector = linear_layer.bias.view(-1, 1)
        # Concatenate into a single vector
        param_vector = torch.cat((weight_vector, bias_vector), dim=0)
        #print(param_vector.size())
        return param_vector

    def forward(self, x, c_logits, w, y_logits):
        """Computes ranking score."""

        #We need to resize, as we have 
        #print(f"Size x: {x.size()}") #[8, ..., ..., ...] # resize to [8, ...]
        #print(f"Size c: {c_logits.size()}") #[8, ...] #no resize needed
        #print(f"Size w: {w.size()}") #[22600, 1] #resize to [8, 22600, 1]
        #print(f"Size y: {y_logits.size()}") #[8, ...] #no resize needed

        # Flatten x and w tensors except the batch dimension
        w_flat = w.view(1, -1).repeat(x.size(0), 1)
        
        # Concatenate along the second dimension (feature dimension)
        flattened = torch.cat((c_logits, w_flat, y_logits), dim=1)
        
        #print(flattened.size())
        return self.lay(flattened)

    def ranking_loss(self, r_f, r_o):
        target = torch.ones_like(r_f)
        #target = torch.Tensor([1]*batch_size)
        
        ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        output = ranking_loss(r_o, r_f, target)
        return output
        #output.backward()
        #return torch.mean(torch.clamp(self.margin + r_f - r_o, min=0))

    def ranking_loss_GPT(self, r_f, r_o):
        """Pairwise ranking loss (Hinge loss)."""
        return torch.mean(torch.clamp(self.margin + r_f - r_o, min=0))

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        if len(batch) > 3:
            competencies = batch[3]
        else:
            competencies = None
        if len(batch) > 4:
            prev_interventions = batch[4]
        else:
            prev_interventions = None
        return x, y, (c, competencies, prev_interventions)


    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch,batch_idx):
        loss = self.run_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, intervention_idxs=None):
        #Step 1: get the batch unpacked
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #Step 2: for the batch, calculate:
        c_sem, c_logits_f, y_logits_f = self.cbm_buggy(x) #_ = c_sem_f
        c_sem, c_logits_o, y_logits_o = self.cbm_oracle(x) #_ = c_sem_o

        #Step 3: get w_f and w_o
        w_f = self.compute_w(self.cbm_buggy).to(device=self.dev)
        w_o = self.compute_w(self.cbm_oracle).to(device=self.dev)
        
        #Step 4: put both inputs to the self fotward
        r_f = self(x, c_logits_f, w_f, y_logits_f)
        r_o = self(x, c_logits_o, w_o, y_logits_o)

        #Step 5: calculate loss and return
        loss = self.ranking_loss(r_f, r_o)
        return loss

    def run_step(self, batch):
        #Step 1: get the batch unpacked
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #Step 2: for the batch, calculate:
        c_sem, c_logits_f, y_logits_f = self.cbm_buggy(x) #_ = c_sem_f
        c_sem, c_logits_o, y_logits_o = self.cbm_oracle(x) #_ = c_sem_o

        #Step 3: get w_f and w_o
        w_f = self.compute_w(self.cbm_buggy).to(device=self.dev)
        w_o = self.compute_w(self.cbm_oracle).to(device=self.dev)
        
        #Step 4: put both inputs to the self fotward
        r_f = self(x, c_logits_f, w_f, y_logits_f)
        r_o = self(x, c_logits_o, w_o, y_logits_o)

        #Step 5: calculate loss and return
        loss = self.ranking_loss(r_f, r_o)
        return loss
        
    def configure_optimizers(self):
        #print("config optimizers")
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay = 4e-05
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }
    
    
class Ranker3(pl.LightningModule):
    def __init__(self, cbm_buggy, cbm_oracle, lr=1e-9, margin=1.0):
        super(Ranker3, self).__init__()
        self.save_hyperparameters(ignore=["cbm_buggy", "cbm_oracle"])
        #super().__init__()
        self.cbm_buggy = cbm_buggy
        self.cbm_oracle = cbm_oracle
        self.n_concepts = cbm_buggy.n_concepts
        self.n_tasks = cbm_buggy.n_tasks
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.margin = margin


        # Fully connected layers
        img_size = 299*299*3
        weight_top_layer = self.n_concepts*self.n_tasks + self.n_tasks
        inp = img_size + self.n_concepts + weight_top_layer
        #print(inp)

        self.lay = torch.nn.Sequential(*[
            torch.nn.Linear(inp, 1),
        ])


    def compute_w(self, model):
        #print("compute_w")
        # Get the last layer (weights w of the model)
        linear_layer = model.c2y_model[0]
        # Flatten weights and biases
        weight_vector = linear_layer.weight.view(-1, 1)
        bias_vector = linear_layer.bias.view(-1, 1)
        # Concatenate into a single vector
        param_vector = torch.cat((weight_vector, bias_vector), dim=0)
        #print(param_vector.size())
        return param_vector

    def forward(self, x, c_logits, w):
        """Computes ranking score."""

        #We need to resize, as we have 
        #print(f"Size x: {x.size()}") #[8, ..., ..., ...] # resize to [8, ...]
        #print(f"Size c: {c_logits.size()}") #[8, ...] #no resize needed
        #print(f"Size w: {w.size()}") #[22600, 1] #resize to [8, 22600, 1]
        #print(f"Size y: {y_logits.size()}") #[8, ...] #no resize needed

        # Flatten x and w tensors except the batch dimension
        x_flat = x.view(x.size(0), -1)
        w_flat = w.view(1, -1).repeat(x.size(0), 1)
        
        # Concatenate along the second dimension (feature dimension)
        flattened = torch.cat((x_flat, c_logits, w_flat), dim=1)
        
        #print(flattened.size())
        return self.lay(flattened)

    def ranking_loss(self, r_f, r_o):
        target = torch.ones_like(r_f)
        #target = torch.Tensor([1]*batch_size)
        
        ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        output = ranking_loss(r_o, r_f, target)
        return output
        #output.backward()
        #return torch.mean(torch.clamp(self.margin + r_f - r_o, min=0))

    def ranking_loss_GPT(self, r_f, r_o):
        """Pairwise ranking loss (Hinge loss)."""
        return torch.mean(torch.clamp(self.margin + r_f - r_o, min=0))

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        if len(batch) > 3:
            competencies = batch[3]
        else:
            competencies = None
        if len(batch) > 4:
            prev_interventions = batch[4]
        else:
            prev_interventions = None
        return x, y, (c, competencies, prev_interventions)


    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch,batch_idx):
        loss = self.run_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, intervention_idxs=None):
        #Step 1: get the batch unpacked
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #Step 2: for the batch, calculate:
        c_sem, c_logits_f, y_logits_f = self.cbm_buggy(x) #_ = c_sem_f
        c_sem, c_logits_o, y_logits_o = self.cbm_oracle(x) #_ = c_sem_o

        #Step 3: get w_f and w_o
        w_f = self.compute_w(self.cbm_buggy).to(device=self.dev)
        w_o = self.compute_w(self.cbm_oracle).to(device=self.dev)
        
        #Step 4: put both inputs to the self fotward
        r_f = self(x, c_logits_f, w_f)
        r_o = self(x, c_logits_o, w_o)

        #Step 5: calculate loss and return
        loss = self.ranking_loss(r_f, r_o)
        return loss

    def run_step(self, batch):
        #Step 1: get the batch unpacked
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #Step 2: for the batch, calculate:
        c_sem, c_logits_f, y_logits_f = self.cbm_buggy(x) #_ = c_sem_f
        c_sem, c_logits_o, y_logits_o = self.cbm_oracle(x) #_ = c_sem_o

        #Step 3: get w_f and w_o
        w_f = self.compute_w(self.cbm_buggy).to(device=self.dev)
        w_o = self.compute_w(self.cbm_oracle).to(device=self.dev)
        
        #Step 4: put both inputs to the self fotward
        r_f = self(x, c_logits_f, w_f)
        r_o = self(x, c_logits_o, w_o)

        #Step 5: calculate loss and return
        loss = self.ranking_loss(r_f, r_o)
        return loss
        
    def configure_optimizers(self):
        #print("config optimizers")
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay = 4e-05
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }
        

class Ranker4(pl.LightningModule):
    def __init__(self, cbm_buggy, cbm_oracle, lr=1e-9, margin=1.0):
        super(Ranker4, self).__init__()
        self.save_hyperparameters(ignore=["cbm_buggy", "cbm_oracle"])
        #super().__init__()
        self.cbm_buggy = cbm_buggy
        self.cbm_oracle = cbm_oracle
        self.n_concepts = cbm_buggy.n_concepts
        self.n_tasks = cbm_buggy.n_tasks
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.margin = margin


        # Fully connected layers
        weight_top_layer = self.n_concepts*self.n_tasks + self.n_tasks
        inp = self.n_concepts + weight_top_layer
        #print(inp)

        self.lay = torch.nn.Sequential(*[
            torch.nn.Linear(inp, 1),
        ])


    def compute_w(self, model):
        #print("compute_w")
        # Get the last layer (weights w of the model)
        linear_layer = model.c2y_model[0]
        # Flatten weights and biases
        weight_vector = linear_layer.weight.view(-1, 1)
        bias_vector = linear_layer.bias.view(-1, 1)
        # Concatenate into a single vector
        param_vector = torch.cat((weight_vector, bias_vector), dim=0)
        #print(param_vector.size())
        return param_vector

    def forward(self, c_logits, w):
        """Computes ranking score."""

        #We need to resize, as we have 
        #print(f"Size x: {x.size()}") #[8, ..., ..., ...] # resize to [8, ...]
        #print(f"Size c: {c_logits.size()}") #[8, ...] #no resize needed
        #print(f"Size w: {w.size()}") #[22600, 1] #resize to [8, 22600, 1]
        #print(f"Size y: {y_logits.size()}") #[8, ...] #no resize needed

        # Flatten x and w tensors except the batch dimension
        w_flat = w.view(1, -1).repeat(c_logits.size(0), 1)
        
        # Concatenate along the second dimension (feature dimension)
        flattened = torch.cat((c_logits, w_flat), dim=1)
        
        return self.lay(flattened)

    def ranking_loss(self, r_f, r_o):
        target = torch.ones_like(r_f)
        #target = torch.Tensor([1]*batch_size)
        
        ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        output = ranking_loss(r_o, r_f, target)
        return output

    def ranking_loss_GPT(self, r_f, r_o):
        """Pairwise ranking loss (Hinge loss)."""
        return torch.mean(torch.clamp(self.margin + r_f - r_o, min=0))

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        if len(batch) > 3:
            competencies = batch[3]
        else:
            competencies = None
        if len(batch) > 4:
            prev_interventions = batch[4]
        else:
            prev_interventions = None
        return x, y, (c, competencies, prev_interventions)


    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch,batch_idx):
        loss = self.run_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch,batch_idx):
        loss = self.run_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, intervention_idxs=None,
    ):
        #Step 1: get the batch unpacked
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #Step 2: for the batch, calculate:
        c_sem, c_logits_f, y_logits_f = self.cbm_buggy(x) #_ = c_sem_f
        c_sem, c_logits_o, y_logits_o = self.cbm_oracle(x) #_ = c_sem_o

        #Step 3: get w_f and w_o
        w_f = self.compute_w(self.cbm_buggy).to(device=self.dev)
        w_o = self.compute_w(self.cbm_oracle).to(device=self.dev)
        
        #Step 4: put both inputs to the self fotward
        r_f = self(c_logits_f, w_f)
        r_o = self(c_logits_o, w_o)

        #Step 5: calculate loss and return
        loss = self.ranking_loss(r_f, r_o)
        return loss

    def run_step(self, batch):
        #Step 1: get the batch unpacked
        #print(len(batch))
        #print(batch[0].size())
        #print(batch[1].size())
        #print(batch[2].size())
        #print(batch[3].size())
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #print(x.size())
        #print(y.size())
        #print(c.size())
        #print(x_org.size())
        #x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        #Step 2: for the batch, calculate:
        c_sem, c_logits_f, y_logits_f = self.cbm_buggy(x) #_ = c_sem_f
        c_sem, c_logits_o, y_logits_o = self.cbm_oracle(x) #_ = c_sem_o

        #Step 3: get w_f and w_o
        w_f = self.compute_w(self.cbm_buggy).to(device=self.dev)
        w_o = self.compute_w(self.cbm_oracle).to(device=self.dev)
        
        #Step 4: put both inputs to the self fotward
        r_f = self(c_logits_f, w_f)
        r_o = self(c_logits_o, w_o)

        #Step 5: calculate loss and return
        loss = self.ranking_loss(r_f, r_o)
        return loss
        
    def configure_optimizers(self):
        #print("config optimizers")
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay = 4e-05
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }
class MetricsAccum:
    def __init__(self):
        self.num_batches = 0
        self.num_batches_g = 0
        self.loss = 0
        self.hits_true = 0
        self.total_true = 0
        self.hits_fake = 0
        self.total_fake = 0
        self.gen_loss = 0

    def disc_accum(self, loss, batch_predicted, batch_label):
        self.num_batches += 1
        self.loss += loss
        self.hits_true += ((batch_predicted == batch_label) * (batch_label == 1)).sum().item()
        self.total_true += (batch_label == 1).sum().item()
        self.hits_fake += ((batch_predicted == batch_label) * (batch_label == 0)).sum().item()
        self.total_fake += (batch_label == 0).sum().item()

    def gen_accum(self, loss):
        self.num_batches_g += 1
        self.gen_loss += loss

    def calculate(self):
        d_loss = self.loss / self.num_batches
        d_accuracy_true = self.hits_true / self.total_true
        d_accuracy_fake = self.hits_fake / self.total_fake
        d_accuracy = (self.hits_true + self.hits_fake) / (self.total_true + self.total_fake)
        g_loss = self.gen_loss / self.num_batches_g
        return Metrics(d_loss, d_accuracy_true, d_accuracy_fake, d_accuracy, g_loss)


class Metrics:
    def __init__(self, d_loss, d_accuracy_true, d_accuracy_fake, d_accuracy, g_loss):
        self.d_loss = d_loss
        self.d_accuracy_true = d_accuracy_true
        self.d_accuracy_fake = d_accuracy_fake
        self.d_accuracy = d_accuracy
        self.g_loss = g_loss

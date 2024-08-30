import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from seqeval.metrics import classification_report
from torchcrf import CRF
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import defaultdict
from seqeval.metrics import classification_report
from seqeval.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import torch

class Conll2003Dataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len):
        """
        Args:
            data_file (string): 路径到CONLL 2003数据集文件。
            tokenizer (BertTokenizer): BERT分词器。
            max_len (int): 序列的最大长度。
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 读取数据并进行预处理
        self.examples = self._load_data(data_file)

    def _load_data(self, data_file):
        """
        从文件中读取数据，并将其转换为模型输入所需的格式。
        """
        examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                line = line.strip()
                if line.startswith('-DOCSTART-') or not line:
                    if words and labels:
                        # 将当前句子添加到示例列表
                        examples.append({
                            'words': words,
                            'labels': labels
                        })
                        words = []
                        labels = []
                    continue
                parts = line.split()
                word, label = parts[0], parts[-1]
                words.append(word)
                labels.append(label)
            if words and labels:
                # 添加文件中最后一句话
                examples.append({
                    'words': words,
                    'labels': labels
                })
        return examples

    def _create_encoding(self, words):
        """
        将单词列表转换为BERT模型的编码。
        """
        encoding = self.tokenizer(
            words,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return encoding

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self._create_encoding(example['words'])
        labels = torch.tensor([self.label_map[label] for label in example['labels']])
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

    def _collate_fn(self):
        """
        自定义collate_fn以处理不同长度的输入。
        """

        def collate_fn(batch):
            batch = default_collate(batch)
            batch['input_ids'] = torch.stack([item[0] for item in batch['input_ids']])
            batch['attention_mask'] = torch.stack([item[1] for item in batch['attention_mask']])
            batch['labels'] = torch.stack([item[2] for item in batch['labels']])
            return batch

        return collate_fn


# 定义标签映射
label_map = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-LOC': 3,
    'I-LOC': 4,
    'B-ORG': 5,
    'I-ORG': 6,
    # 根据需要添加更多标签
}

# 使用示例
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = Conll2003Dataset('train.txt', tokenizer, max_len=128)

# 定义模型
class NERModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(NERModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = -self.crf(emissions=logits, tags=labels, mask=attention_mask)
        return logits, loss

# 实例化模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = NERModel('bert-base-uncased', num_labels=你的标签数量)

# 准备数据加载器
train_dataset = Conll2003Dataset(tokenizer, 'train.txt', max_len=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
for epoch in range(你的训练轮数):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {
            'input_ids': batch['input_ids'].squeeze(1),
            'attention_mask': batch['attention_mask'].squeeze(1)
        }
        labels = batch['labels'].squeeze(1)
        logits, loss = model(inputs['input_ids'], inputs['attention_mask'], labels=labels)
        loss.backward()
        optimizer.step()

# 评估模型
def evaluate(model, data_loader, device):
    """
    评估模型性能。

    参数:
        model (nn.Module): 要评估的模型。
        data_loader (DataLoader): 包含验证集或测试集的DataLoader。
        device (str): 模型和数据应该运行的设备，例如 'cuda' 或 'cpu'。
    """
    model.eval()  # 将模型设置为评估模式
    predictions, true_labels = [], []

    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        for batch in data_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)

            # 获得模型输出
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            logits = outputs[0]

            # 使用CRF解码器或argmax获取预测标签
            predictions.extend(model.crf.decode(logits, inputs['attention_mask']).tolist())
            true_labels.extend(labels.tolist())

    # 转换预测和真实标签为正确的格式
    predictions = [[label_list[i] for label_list in predictions] for i in range(len(predictions[0]))]
    true_labels = [[label_list[i] for label_list in true_labels] for i in range(len(true_labels[0]))]

    # 计算评估指标
    report = classification_report(true_labels, predictions, digits=4)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    print("Classification Report:\n", report)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return precision, recall, f1

# 保存模型
# torch.save(model.state_dict(), 'ner_model.pth')
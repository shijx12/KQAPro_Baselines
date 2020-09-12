import torch
import torch.nn as nn

from utils.BiGRU import BiGRU

class GRUClassifier(nn.Module):
    def __init__(self, vocab, dim_word, dim_hidden):
        super().__init__()
        
        num_class = len(vocab['answer_token_to_idx'])
        num_words = len(vocab['word_token_to_idx'])

        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.3)
        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=2, dropout=0.2)

        self.classifier = nn.Sequential(
                nn.Linear(dim_hidden, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_class)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, questions):
        """
        Args:
            - questions (LongTensor) [bsz, max_len]
        """
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        # print(question_lens)
        question_input = self.word_dropout(self.word_embeddings(questions))
        _, question_embeddings, _ = self.question_encoder(question_input, question_lens)
        logits = self.classifier(question_embeddings)
        return logits

from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
import numpy as np
import csv

class BertTagger:
    def __init__(self):
        self.definitions = self.read_definitions("./data/definitions.tsv")
        model_name = 'bert-base-uncased'
        config = BertConfig.from_pretrained(model_name)
        config.output_hidden_states = False
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
        self.model = BertForSequenceClassification.from_pretrained("model_save")

        self.preps = ["across", "to", "on", "with", "in", "of", "at", "inside", "during", "from", "as", "through", "for", "along", "like", "about", "into", "towards", "down", "behind", "round", "before", "by", "against", "between", "onto", "off", "beside", "around", "over", "among", "above", "after", "beneath"]

    def read_definitions(self, path):
        with open(path) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            defs = dict(reader)
        return defs

    def tag(self, sentence):
        
        words = sentence.split(" ")
        for idx, word in enumerte(words):
            if word in self.preps:
                temp = words[idx]
                words[idx] = "<head>"+temp+"</head>"
                prepared = " ".join(words)
                print(prepared)




        data = self.tokenizer(
            text=[sentence],
            add_special_tokens=True,
            max_length=100,
            truncation=True,
            padding=True, 
            return_tensors='pt',
            return_token_type_ids = False,
            verbose = True)

        logits = self.model(data['input_ids'], token_type_ids=None, attention_mask=data['attention_mask'])[0].detach().numpy()
        prediction = self.definitions[str(np.argmax(logits))]
        print("sentence : {}\nprediction : {}\n\n".format(sentence, prediction))
        return prediction 


tagger = BertTagger()
# IN
tagger.tag("I am <head>in</head> big trouble")
tagger.tag("I am <head>in</head> a big airplane")
tagger.tag("I am <head>in</head> New York")
tagger.tag("<head>In</head> 2020 the president of the United States will be elected")
tagger.tag("I always see you <head>in</head> my dreams.")
tagger.tag("I am speaking <head>in</head> portuguese.")
tagger.tag("The president appears <head>in</head> a weird manner.")

# BY
tagger.tag("The Crisis was handled <head>by</head> him.")
tagger.tag("He is leading the polls <head>by</head> far.")

# WITH
tagger.tag("He is swimming <head>with</head> his hands.")
tagger.tag("She blinked <head>with</head>  confusion.") # Manner
tagger.tag("He washed a small red teacup <head>with</head>  water.")  # Means

# ON
tagger.tag("My friend is <head>on</head> the way to Moscow.")
tagger.tag("When she was a little girl people saw unrealistic cowboy films <head>on</head> television")
tagger.tag("I am <head>on</head> my way to New York.")

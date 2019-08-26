import requests
import data_loader
import pandas as pd
import time
URL = 'http://api.conceptnet.io/relatedness?node1=/c/en/{}&node2=/c/en/{}'
GENERATED_SENTENCE_FILE_PATH = 'out/generated_sentences-5.7.0.txt'
OUTPATH = 'out/generated_sentence_ranking.csv'
def get_relatedness(subject, object_):
    try:
        res = requests.get(URL.format(subject.replace(' ', '_'), object_.replace(' ', '_'))).json()
        time.sleep(2)
    except Exception as e:
        print(subject, object_)
        return None
    return res['value']

def main():
    data = data_loader.load_normalized_dataset()
    relations = data['relation'].unique().tolist()
    results = []
    with open(GENERATED_SENTENCE_FILE_PATH, 'r') as f:
        for ind, line in enumerate(f.readlines()):
            print(ind)
            if ind%2==1:
                results[ind//2]['sentence'] = line.strip()
            if ind%2==0:
                results.append({})
                for relation in relations:
                    if relation in line:
                        subject, object_ = line.split(relation)
                        val = get_relatedness(subject, object_.strip())
                        results[ind//2]['subject'] = subject
                        results[ind//2]['object'] = object_.strip()
                        results[ind//2]['score'] = val

        results = pd.DataFrame(results)
        results.sort_values('score').reset_index().to_csv(OUTPATH)

if __name__ == '__main__':
    # main()
    data = data_loader.load_normalized_dataset()
    sample = data.sample(1000)
    sample['score'] = sample.apply(lambda x: get_relatedness(x['subject'], x['object']), axis=1)
    sample.to_csv('out/conceptnet_sentences.csv')

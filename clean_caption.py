import json
import pandas as pd
import os
import re

cleaned_text = []

def create_df(output='./data/image_captions.csv'):    
    df2 = pd.read_csv('./data/indiana_projections.csv')
    df1 = pd.read_csv('./data/indiana_reports.csv')
    
    images_captions_df = pd.DataFrame({'image': [],
                                    'caption': [],
                                    'indication': [],
                                     'problems': []})
    for i in range(len(df2)):
        uid = df2.iloc[i]['uid']
        image = df2.iloc[i]['filename']
        index = df1.loc[df1['uid'] ==uid]
        
        if not index.empty:    
            index = index.index[0]
            caption = 'Findings: ' + str(df1.iloc[index]['findings']) + '\n Impression: ' + str(df1.iloc[index]['impression'])
            indication = df1.iloc[index]['indication']
            problems = df1.iloc[index]['Problems']
            if type(caption) == float:
            
                continue 
            images_captions_df = pd.concat([images_captions_df, pd.DataFrame([{'image': image, 'caption': caption,
                                    'indication': indication,'problems': problems}])], ignore_index=True)
    
    images_captions_df.to_csv(output)
    
    return images_captions_df

def check_similar_samples(df, text):
    #print(len(df), " : " + text)
    print_count = 0
    for caption in df['caption']:
        if print_count <= 15:
            for sent in caption.split(". "):
                if text in sent and "XXXX" not in sent:
                    print(sent)
                    print_count += 1
                    #print()
                elif text in sent:
                    print(sent)
                    print_count += 1
                    #print("in elif")
    print()
    return

def clean_convert_numbers(text):
    text = text.replace('PA and lateral views were obtained.', "")
    text= re.sub('\d+. ','\n - ', text)
    result = []
    result.extend(re.findall('\d+ cm', text))
    result.extend(re.findall('\d+mm', text))
    result.extend(re.findall('\d+ mm', text))
    if len(result) > 0:
        for each in result:
            if each.endswith('mm'):
                num = each.split('mm')[0]
                text = text.replace(each, '<1 cm')
            else:
                num = each.split('cm')[0]
                if int(num) == 0:
                    text = text.replace(each, "<1 cm")
    return text

def apply(images_captions_df, unclean_sent, clean_sent, indices, df_output= './data/image_captions.csv'):
    assert len(unclean_sent) == len(clean_sent)
    print(f"Applying correction to {len(indices)} caption(s).")
    for ind in indices:
        caption = images_captions_df.iloc[ind]["caption"]
        for i , each in enumerate(unclean_sent):
            # print(each)
            # print(clean_sent[i])
            # print()
            if clean_sent[i].endswith(".") or clean_sent[i].endswith(". "):
                caption = caption.replace(each, clean_sent[i])
            else:
                caption = caption.replace(each, clean_sent[i] +".")
        
        images_captions_df.loc[ind,"caption"] = caption
        print(images_captions_df.iloc[ind]['caption'])
        print()
    
    images_captions_df.to_csv(df_output)
    
    return images_captions_df
    
def findall_occurrences(df, ind):
    #indices = [ind]
    indices = list(df[df['caption']== df.iloc[ind]['caption']].index)
    print(f"Found {len(indices)} with the same captions")
    print()
    return indices


def split_undone(captions, size=1200):
    data = list(captions.items())
    count=0
    for batch in range(0, len(data), size):
        with open(f"./data/data_shard/unclean_caption{count}.json", "w") as f:
            json.dump(dict(data[batch: batch + size]), f)
        count += 1
    return

def clean_text(images_captions_df, df_output= './data/image_captions.csv', split=False, load_from_file=True):
    
    if os.path.exists("./data/clean_caption_index.txt"):
        with open("./data/clean_caption_index.txt","r") as f:
            indices = f.readlines()
        clean_caption_index = []
        for each in indices:
            clean_caption_index.append(int(each))
    else:
        clean_caption_index = []
        
    total = len(images_captions_df)
    # Get all captions that have XXXX in them
    if load_from_file:
        print(os.listdir("./data/data_shard")[0])
        with open(os.path.join(os.path.abspath('./data/data_shard'),os.listdir("./data/data_shard")[0]), 'r') as f:
            contains_xxx = json.load(f)
        count = len(contains_xxx) - len(clean_caption_index)
    else:
        contains_xxx = {}
        count = 0
        for ind, each in enumerate(images_captions_df['caption']):
            if "XXXX" in each:
                count += 1
                unclean_sent = []
                for sent in each.split(". "):
                    if "XXXX" in sent:
                        unclean_sent.append(sent)
                        
                contains_xxx[ind] = unclean_sent
    
    print(f"There are {count} unclean captions out of {total} left...")
    print()
    
    if split:
        split_undone(contains_xxx)
        print("split done..")
    
    
    field="image"
    end = False   
    for ind, unclean_text in contains_xxx.items():
        ind = int(ind)
        if len(clean_caption_index) > 0 and ind in clean_caption_index:
            continue
        
        clean_text = []
        print(f"There is/are {len(unclean_text)} redacted text in this caption..")
        print()
        for i, sentence in enumerate(unclean_text):
            if "XXXX" in sentence:
                print(f"Image - {ind} , Image name {images_captions_df[field][ind]}, Text - ({i})\n {sentence}.")
                print()
                correction = input(f"Type correction / deletions or match a phrase with the database below...\n")
                
                if correction == "exit":
                    end = True
                    break
                if correction == "ignore":
                    continue
                
                if correction == "check":
                    print(images_captions_df.iloc[ind]['caption'])
                    print()
                    print(f"Image - {ind}, Text - ({i})\n {sentence}.")
                    print()
                    correction = input("Type correction / deletions or match a phrase with the database below...\n")
                
                if correction == "ignore":
                    continue
                    
                if "match-" in correction:
                    text = correction.split("-")[1]
                    continue_match = True
                    just_checked = False
                    while continue_match:
                        #print("I am inn match.")
                        if not just_checked:
                            check_similar_samples(images_captions_df, text)
                        correction = input('Type correction / deletions or match a phrase with the database below..\n')
                        if "match-" in correction:
                            #print("I am in the while loop.")
                            text = correction.split("-")[1]
                            just_checked = False
                            continue
                        elif correction == 'check':
                            print(images_captions_df.iloc[ind]['caption'])
                            print()
                            print(f"Image - {ind}, Text - ({i})\n {sentence}.")
                            print()
                            just_checked = not just_checked
                        else:
                            continue_match = False
                
                correction = correction.split("/")
                additions = correction[0].split(',')
                clean_sentence = None
                
                for x in additions:
                    #print(x)
                    if clean_sentence is None:
                        clean_sentence = sentence.replace("XXXX", x, 1)
                    else:
                        clean_sentence = clean_sentence.replace("XXXX", x, 1)
                        
                    print(clean_sentence)
                    print()
                    
                if len(correction) > 1:
                    if correction[1] == "all":
                        clean_sentence = ""
                    else:
                        deletions = correction[1].split(";")[:-1]
                        print(deletions)
                        for x in deletions:
                            #print(x)
                            if "=" in x:
                                old_text = x.split("=")[0].strip(" ")
                                new_text = x.split("=")[1].strip(" ")
                                #
                                #print("in = in deletions")
                                if old_text != " ":
                                    clean_sentence = clean_sentence.replace(old_text, new_text,1)
                            else:
                                #print('yes')
                                if x != "" or x != " ":
                                    #print("this:",x)
                                    x =x.strip(" ")
                                    clean_sentence = clean_sentence.replace(" " + x ,"",1)
                
                print(clean_sentence)
                print()
                clean_text.append(clean_sentence)
                                
        if end:
            images_captions_df.to_csv(df_output)
            return
            
        # find all occurrences of unclean text
        indices = findall_occurrences(images_captions_df, ind)
        clean_caption_index.extend(indices)
        
        with open("./data/clean_caption_index.txt", 'w') as f:
            for each in clean_caption_index:
                f.write(f"{each}\n")
            
        # apply corrections to all occurrences
        images_captions_df = apply(images_captions_df, unclean_text, clean_text, indices)
        print("Saving in file now..")
        images_captions_df.to_csv(df_output)
       
    print('Saving file now...')
    images_captions_df.to_csv(df_output)
    
    return



if __name__ == '__main__':
    df = pd.read_csv('./data/image_captions.csv') #create_df()
    clean_text(df)
    
    # TODO
    # Convert all float to integers, convert all float areas to int areas (1.9cm x 1.8cm)
    # add subcentimeter to size
    #change "The lungs are intact " to bony structures are intact.
    #two, multiple
    # Findings: The XXXX examination consists of frontal and lateral radiographs of the chest.
    #The chest examination consists of frontal and lateral radiographs of the chest
    #These findings and recommendations were discussed XXXX
    #by Dr.
    #telephone at XXXX p.m.
    #technologist receipt of the results,  XXXX T6, nan, 
    #Breast implants there is a moderate wedge around deformity of the midthoracic vertebrae, XXXX T6, age-indeterminate.
    
    #. Well-expanded and clear lungs. Mediastinal contour within normal limits. No acute cardiopulmonary abnormality identified.
    #which has increased in size compared to prior chest radiograph and
    
    #Impression: 1. Round area of density measuring 1.9 x 1.8 cm in left superior lower lobe with interval increased size compared to prior 
    #imaging. Recommend XXXX chest, abdomen and pelvis with contrast for further evaluation. Dr. XXXX XXXX notified by the Veriphy 
    # critical result notification XXXX of the left pulmonary mass and recommended followup XXXX chest, abdomen and pelvis with 
    # contrast at XXXX XXXX/XXXX.
    
    
    #recommend a XXXX chest, abdomen and pelvis with contrast as this area is suspicious for potential malignancy
    #Deformity of the right clavicletoseen
    # The scarring in the left lower lobe is again noted and unchanged from prior exam.
    #This measures 3.2 cm at the level the right apex.
    # Compared to prior exam, there is XXXX prominence of the mediastinal contour near the right hilum.
    # Calcified mediastinal and hilar lymph XXXX are consistent with prior granulomatous disease
    #If there is concern for soft tissue bone or bony abnormality of the thorax
    # Findings and recommendations were discussed XXXX. XXXX in the XXXX department at XXXX a.m. XXXX/XXXX.
    #&gt;]
    #noted on recent PA chest radiograph
    # Comparison XXXX, XXXX.
    # ijn: internal jugular vein, SVC: superior vena cava, PICC: peripherally inserted central catheter
    #Stable short segment catheter tubing overlying the left XXXX, XXXX to reside within anterior chest soft tissues on recent chest CT
    #Unchanged
    #unchanged in the interval
    #osseous structures - bony structures
    #PA and lateral chest radiograph may be of benefit XXXX clinically feasible..
    # identified status post thoracentesis; pleural effusion,d   to; Findings:   radiographs.
    # Dr. XXXX- XXXX was called and informed of these critical results at XXXX.
    #compared to prior study; on the frontal view, Findings:   . ;XXXX XXXX notified of the critical results at XXXX on XXXX, XXXX by telephone and acknowledged receipt of these results..
    #No significant interval change compared to prior study; No, no interstitial infiltrates noted.
    #prior studies
    #remove extra '.'; The following examination consists of frontal and lateral radiographs of the chest;
    #A total of 3 images were obtained.;The following examination consists of frontal, lateral and swimmers lateral radiographs of the thoracic spine
    # chck;  lateral chest examination was obtained
    # at is not identified on the PA projection.;  Calcificationseen
    #as more readily demonstrated on the previous CT chest study from 
    #Stableprominent; and below the  -of-view; certified radiologist;If there are any questions about this examination please
    #XXXX for the opportunity to assist in the care of your patient; check are normal
    #stable from prior exam.
    # attributed to the patient's recent abdominal surgery
    
    
    #Findings: There are no focal airspace opacities within the lungs. There is a 1 cm nodular density projecting in the right midlung between the third and fourth right anterior ribs   . To the pulmonary interstitium is not clear, making it the vasculature somewhat indistinct in the mid and lower lungs. This may reflect multiple parenchymal nodules. . Mediastinal contours appear grossly normal. There are small calcified left hilar lymph nodes. The heart and pulmonary vasculature otherwise appear normal. Pleural spaces appear clear.
    #Impression: 1. A 1 cm nodular density seen  projecting in the right midlung. Recommend noncontrasted enhanced CT chest for 
    # evaluation of this nodule. Does this patient have known risk factors for malignancy? 2. Somewhat indistinct pulmonary interstitium
    # possibly reflecting underlying pulmonary sarcoidosis
    # Findings: The chest examination consists of frontal and lateral radiographs of the chest
    # Findings: The chest examination consists of frontal and lateral radiographs of the chest
    # Stable appearance of the chest.
    # Stable appearance of the chest.
    # Findings: The chest examination consists of frontal and lateral radiographs of the chest
    # Findings: The chest examination consists of frontal and lateral radiographs of the chest
    # Findings: Frontal and lateral views of the chest show an unchanged cardiomediastinal silhouette
    # Findings: Frontal and lateral views of the chest show an unchanged cardiomediastinal silhouette
    # Findings: Frontal and lateral views of the chest show normal size and configuration of the cardiac silhouette
    
    #     Image - 3903, Text - (3) -Add it back
    #  Visualized XXXX of the chest XXXX are within normal limits.
    #  Impression: 2 cm noncalcified nodule in the right lower lobe would benefit from a XXXX..
    
    #  Impression: Patchy airspace disease on the lateral view, probably within the right lower lobe, XXXX a pneumonia superimposed on XXXX severe underlying emphysema. 
    #  Recommend following this process to resolution.
    #automated implantable cardioverter defibrillator (AICD) ; 9th, fourth
    #Comparison;  on lateral view, T9,T6, T8, T11 and T12
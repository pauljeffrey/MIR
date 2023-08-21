import json
import pandas as pd
import os
import re
import math
import openai
import random
import string  as st
from Questions import QUESTION_TEMPLATE

characters = st.ascii_letters + st.digits + st.punctuation
openai.api_key = "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" #os.getenv("OPENAI_API_KEY")


def return_report_template(report):
    system_message = f"{report}\nYou are a consultant radiologist with experience reporting chest radiographs. Edit this report above, \
        correct all errors .  Provide the most likely radiological term for the placeholder 'xx'. Clues and answers for this placeholder \
        may be found in the report itself. the placeholder can be removed but resulting sentence must be coherent. Do not remove \
        numbers. Report only the findings seen in this report. if you note a reference or comparison to another report, remove it. \
        Convert abbreviations to their full meanings if possible. only return the corrected version of the report in the same structure.\
        Do not make additional statements.Be assertive.Do not repeat any sentence.Examples: defibrillator generator xx xx overlying tip=defibrillator generator with leads overlying tip, \
        xx opacity xx atelectasis =basilar patchy opacity suggests atelectasis, areas of xx scarring= areas of scarring."

    return system_message

# def return_question_template(report, questions):
#     system_message =f"You are a radiologist. with this chest x-ray report below: {report} \nAnswer the following questions: {questions}\n \
#    Return your answers in the order that the questions were asked. if any question is not applicable based on the report provided, \
#     state that it is not.\
#     Be assertive in your answers.Don't reference this report in your answers.Do not use the word 'report' or 'mention' in your answers. Answer all questions."
    
#     return system_message

# def return_question_template(report, questions):
#     system_message =f"You are a radiologist. You are given the chest x-ray report as follows: {report} \nUse the report to Answer the \
#     following questions: {questions}\n Return your answers in the order that the questions were asked. \
#     Be assertive in your answers. when answering a question, treat the report like its the xray image itself.Return the best answer possible given \
#     the information in the report.Answer all questions exactly like this: 1.answer, 2.answer \
#     Make no additional statement or sentence."
    
#     return system_message

def return_question_template(report, questions):
    system_message =f"You are aconsultant radiologist. You are given the report of a chest xray image as follows: {report} \nImagine that you are \
    looking at the chest xray image directly instead of this report and try to answer the \
    following questions: {questions}\n Return your answers in the order that the questions were asked. \
    Be assertive in your answers and dont use the phrase 'Not applicable' in your answers. Remember you are looking at the image itself.Answer all questions exactly like this: 1.answer, 2.answer \
    Make no additional statement or sentence."
    
    return system_message
#Return the best answer possible given the information in the report.

def return_indication_template(indication):
    system_message  = f"{indication}.This statement was used as a clinical indication for a chest xray. you are a medical doctor.Correct all the \
        errors in it and return only the meaningful key words in it. Convert abbreviations to full meanings.\
        Don't mention 'indication' or 'chest xray' in your response.Return 'no indication' if the statement makes no sense. tb means tuberculosis"
    return system_message

def return_problems_template(problems):
    system_message = f"{problems}.The following findings were noted on a chest xray by a radiologist. correct all errors and return only the key words that are present in the findings. \
         Return 'normal study' if the findings are normal. Return the findings verbatim if you can't make sense of it. Do not make extra comments or add extra key words"
    return system_message

def create_df(output='./data/image_captions.csv'):    
    df2 = pd.read_csv('./data/indiana_projections.csv')
    df1 = pd.read_csv('./data/indiana_reports.csv')
    images_captions_json = []
    images_captions_df = pd.DataFrame({'image': [],
                                    'caption': [],
                                    'indication': [],
    
                                     'problems': []})
    problems_list = ""
    #check = [True, True]
    for i in range(len(df2)):
        uid = df2.iloc[i]['uid']
        image = df2.iloc[i]['filename']
        index = df1.loc[df1['uid'] ==uid]
        
        if not index.empty:    
            index = index.index[0]
            problems_list+= str(df1.iloc[index]["Problems"]).lower()
            problems = df1.iloc[index]["Problems"] if (str(df1.iloc[index]["Problems"]) != "nan" or "normal" not in df1.iloc[index]["Problems"].lower()) else "none"
            if problems != "none":
                problems = problems.replace(",","; ").replace("/","; ").replace(":",";").replace(".","; ").replace(";","; ").lower()
            #caption = ""    
            if (str(df1.iloc[index]['findings']) == 'nan' or df1.iloc[index]['findings'] == "")  and (str(df1.iloc[index]['impression']) != "nan" or df1.iloc[index]['impression'] != ""):
                impression = str(df1.iloc[index]['impression'])
                if re.search("\d+. ",impression):
                    search_index = re.search("\d+. ", impression).span()[0]
                    caption = 'Findings: ' + impression[:search_index+1] + ".\nProblems: " + problems + '.\nImpression: ' + impression[search_index:]
                    # if check[0]:
                    #     print(caption, uid, image)
                    #     check[0] = False
                else: 
                    impression = impression.replace('.',' . ').split(' . ')
                    caption = "Findings:" + " . ".join(impression[:-2]) + ".\nProblems: " + problems +  ".\nImpression: " + " . ".join(impression[-2:])
            
            elif (str(df1.iloc[index]['findings'] != 'nan') or df1.iloc[index]['findings'] != "") and (str(df1.iloc[index]['impression'] == 'nan') or df1.iloc[index]['impression']==""):
                findings = str(df1.iloc[index]['findings'])
                if re.search("\d+. ",findings):
                    search_index = re.search("\d+. ", findings).span()[0]
                    caption = 'Findings: ' + findings[:search_index+1] + ".\nProblems: " + problems + '.\nImpression: ' + findings[search_index:]
                    # if check[1]:
                    #     print(caption, uid, image)
                    #     check[1] = False
                else: 
                    findings = findings.replace('.',' . ').split(' . ')
                    caption = "Findings: " + " . ".join(findings[:-2]) + ".\nProblems: " + problems +  ".\nImpression: " + " . ".join(findings[-2:])
            
            elif (str(df1.iloc[index]['findings'] == 'nan') or df1.iloc[index]['findings'] == "") and (str(df1.iloc[index]['impression'] == 'nan') or df1.iloc[index]['impression']==""):
                continue
            else:
                caption = 'Findings: ' + str(df1.iloc[index]['findings']) + ".\nProblems: " + problems + '.\nImpression: ' + str(df1.iloc[index]['impression'])
                
            indication = str(df1.iloc[index]['indication']).lower().replace("xxxx","") if str(df1.iloc[index]['indication'])  else "none"
            # if type(caption) == float:
            #     continue 
            caption = clean_convert_numbers(caption.lower().replace("xxxx","xx"))
            
            images_captions_df = pd.concat([images_captions_df, pd.DataFrame([{'image': image, 'caption': caption,
                                    'indication': indication,'problems': problems}])], ignore_index=True)
            images_captions_json.append({"image":image, "caption":caption, "indication": indication, "problems": problems})
            
    print(f"There are {len(images_captions_df)} in the dataset. ")
    #print(set(list(problems_list)))
    
    images_captions_df.to_csv(output)
    #images_captions_df.to_json("./data/images_captions.json")
    with open("./data/images_captions.json","w") as f:
        json.dump(images_captions_json,f)
    
    return #images_captions_df

def clean_convert_numbers(text):
    text = text.lower()
    if "12/33" in text:
        text = text.replace("12/33", "36 %")
    # for each in text.split("."):
    #     print(each)
    # print(text.split("impression:")[0])
    # print(text.split("impression:")[1])
    text = text.split("impression:")[0] + "impression: " + re.sub('\d+. ','\n - ',text.split("impression:")[1])
    text = re.sub("\d+/\d+/x*","", text)
    text = re.sub('pa and lateral views were obtained.|lateral views of the chest were obtained on', "", text)
    text = re.sub('pa and lateral views were obtained on|[a-z]+\s*[and]+\d*lateral views of the chest were obtained on', "", text)
    text = text.replace("no comparison chest x-xxxx","")
    #text = text.replace()
    text  = text.replace("xxxx",'xx')
    text = re.sub("the xx examination consists of frontal and lateral radiographs of the chest|&gt;","",text)
    text = re.sub("the chest examination consists of frontal and lateral radiographs of the chest|and lateral chest examination was obtained", '',text)
    text = re.sub("these findings and recommendations were discussed xx","",text)
    
    text = re.sub("lateral\s*chest\s*examination\s*was\s*obtained|was\s*obtained",'',text)
    text = re.sub("comparison|was obtained","",text)
    text = re.sub("comparison xx","",text)
    text = re.sub("comparison xx, xx","",text)
    text = re.sub("Findings and recommendations were discussed xx. xx in the xx department at xx a.m. xx/xx", "",text)
    text = re.sub("compared to prior|noted on recent pa chest radiograph|prior study|mildly increased","", text)
    text = re.sub("compared to prior|noted on recent pa chest radiograph|lateral chest radiograph|frontal chest radiograph","", text)
    text = re.sub("compared to prior|noted on recent pa chest radiograph|lateral chest radiograph|frontal chest radiograph","", text)
    text = re.sub("prior studies|prior study|compared to|previous|recent|pa and lateral chest radiograph [^may]|pa and lateral chest radiograph", "",text)
    text = re.sub("examinations*|study|lateral view|frontal view|pa view|compared to prior exam|compared", "",text)
    text = re.sub("from old", "", text)
    text = re.sub("unchanged in the interval|unchanged|in the interval|interval|on the frontal view|on the lateral view", "",text)
    text = re.sub("acknowledged receipt|no significant interval change compared to prior study|significant interval change|\s+interval change|prior studies|compared to|compared|A total of \d+ images were obtained", "",text)
    text = re.sub("The following examination consists of frontal, lateral and swimmers lateral radiographs of the thoracic spine|\
        \s+is not identified on the pa projection|on the pa projection|not identified on the pa projection","",text)
    text = re.sub("as more readily demonstrated on the previous CT chest study from|below the [a-z\s\d]+ -of-view|-of-view","",text)
    text = re.sub("xx-a-xx|for the opportunity to assist in the care of your patient|check are normal|stable from prior exam","",text)
    text = re.sub("prior exam|views|view","",text)
    text = re.sub("have\s*increased\s*in\s*size\s*and\s*number|have\s*increased\s*in\s*size|have\s*increased\s*in\s*number","",text)
    text = re.sub("have\s*increased","",text)
    text = re.sub("midline\s*sternotomy\s*x+", "midline sternotomy wires", text)
    text = re.sub("x+\s*sternotomy", "midline sternotomy", text)
    text = re.sub("cardiac\s*x+\s*generator",'cardiac defibrillator generator', text)
    text = re.sub("x+\s*x+\s*are\s*normal|x+\s*x*\s*are\s*intact", "bony structures and soft tissues are intact",text)
    text = re.sub("x+\s*are\s*normal|x*\s*are\s*intact", "osseous structures are intact",text)
    text = re.sub("xx scarring","suggests scarring",text)
    text = re.sub("x+\s*sternotomy\s*x+","midline sternotomy wires", text)
    text = re.sub("no\s*x+\s*pleural\s*effusion", "no evidence of pleural effusion", text)
    text = re.sub("attributed\s*to\s*the\s*patient's\s*recent\s*abdominal\s*surgery","probably due to a recent surgery", text)
    text = re.sub("AICD","automated implantable cardioverter defibrillator (AICD)", text)
    text = re.sub("pulmonary\s*xx",'pulmonary vasculature',text)
    text = re.sub("xx\s*deformity","wedge deformity",text)
    text = re.sub("recommend\s*a\s*x+\s*scan", "recommend a ct scan", text)
    text = re.sub("xx\s*sternotomy\s*xx","midline sternotomy sutures",text)
    text = re.sub("surgical\s*x+","surgical clips",text)
    text = re.sub("svc","superior vena cava (svc)", text)
    text = re.sub("ivc", "inferior vena cava (ivc)", text)
    text = re.sub("central\s*x+\s+xx|central\s*venous\s*xx|central\s*x+\s*catheter", "central venous catheter", text)
    text = re.sub("x+\s+x+\s*are\s*intact|the\s*x+\s+x+\s*are intact", "the bony structures and soft tissues are intact",text)
    text = re.sub("x+\s+x+\s*are\s*normal|the\s*x+\s+x+\s*are normal", "the bony structures and soft tissues are normal",text)
    text = re.sub("kv","kilovoltage", text)
    text = re.sub("x+\s*opacity,\s*x+\s*atelectasis","patchy opacity, suggest atelectasis",text)
    text = re.sub("x+\s*opacity,\s*x+\s*subsegmental atelectasis","patchy opacity, suggest subsegmental atelectasis",text)
    text = re.sub("x+\s*opacity,\s*x+\s*scar","patchy opacity, suggest scar", text)
    text = re.sub("x+\s*opacity,\s*x+\s*atelectasis","patchy opacity, suggest atelectasis",text)
    text = re.sub("x+\s*airspace\s*disease","patchy airspace disease",text)
    text = re.sub("picc","peripherally inserted central catheter (picc)",text)
    text = re.sub("hilars*lymphs*x*|hilar\s*x*\s+x+","hilar lymph nodes", text)
    text = re.sub("lymph\s*xx","lymph nodes", text)
    text =re.sub("basilar\s*airspace\s*disease\s*x*\s*atelectasis","basilar airspace disease suggests atelectasis",text)
    text = re.sub("aortic\s*x+","aortic arch",text)
    text = re.sub("minimal\s*x+airspace\s*opacity","minimal patchy airspace opacity", text)
   
    text = re.sub('cardiac\s*xx\s*generator','cardiac defibrillator generator',text)
    text = re.sub("xx\s*and\s*soft\s*tissues\s*are\s*unremarkable|x+\s*x*\s*are\s*unremarkable", "bony structures and soft tissues are unremarkable", text)
    text = re.sub("costophrenic\s*xx", "costophrenic angle", text)
    text = re.sub("xx\s*of\s*[a-z]*\s*costophrenic [a-z]*","blunting of the costophrenic angle",text)
    text = re.sub("xx\s*calcific\s*density|xx\s*density","increased calcified density", text)
    text = re.sub("xx\s*fissures","pleural fissures", text)
    text = re.sub("are\s*again\s*noted","are noted", text)
    
    text = re.sub("findings:\s*x+,*\s*x+","findings:",text)
    text = re.sub("obscured\s*heart\s*x+","obscured heart silhouette", text)
    
    
    result = []
    
    result.extend(re.findall('\d+\s*cm|\d+.\d+\s*cm|\d+\s*mm|d+.\d+\s*mm|\d+\s*centimeter|\d+.\d+\s*centimeter|\d+\s*centimetre|\d+.\d+\s*centimetre|\d+\s*subcentimeter|\d+.\d+\s*subcentimeter|\d+\s*millimeter|\d+.\d+\s*millimeter|\d+\s*millimetre|\d+.\d+\s*millimetre|\d+.\d+', text))

    #1.6cm x 1 centimeter, 10mm 2.5 cm 1.9 x 1.8 cm
    #12 sept
    if len(result) > 0:
        for each in result:
            #print(each)
            if each.endswith('mm') or each.endswith("millimeter") or each.endswith('subcentimeter'):
                num = each.split('mm')
                #print(len(num))
                if len(num) == 1:
                    num = each.split('millimeter')
                    #print('millimeter')
                if len(num) == 1:
                    num = each.split('subcentimeter')
                    #print('subcentimeter')
                    
                #print(num)
                num=num[0]
                num = re.sub("-\s*\d+","",num)
                start, end  = int(re.search("\d+",num).span()[0]) , int(re.search("\d+",num).span()[1])
                #print(type(start), type(end))
                num = num[start:end]
                #print(num)
                num = float(num)/10
                
                if num >= 1.0:
                    text = text.replace(each, str(round(num)) + " cm")
                else:
                    text = text.replace(each, '<1 cm')
            else:                
                num = each.split('cm')
                #print(num)
                if len(num) ==1:
                    num = each.split('centimeter')
                    #print(num,'centimeter')
                elif len(num) ==1:
                    #print('centimetre')
                    num  = each.split('centimetre')
                
                num = num[0].strip()
               
                #num=num[0]
                if "-" in num or "*" in num or "_" in num:
                    num = re.sub("-\s*\d+","",num)
                    #print(num)
                    start, end  = int(re.search("\d+",num).span()[0]) , int(re.search("\d+",num).span()[1])
                    #print(type(start), type(end))
                    num = num[start:end]
                #print(num)
                #print(num)
                
                if round(float(num)) == 0:
                    text = text.replace(each, "<1 cm")
                else:                  
                    #print(round(float(num)))
                   
                    text = text.replace(each, str(round(float(num.strip()))) + " cm")
    
    text = text.replace('.'," . ")
    sentences = text.split(" . ")
    cleaned_text  = []
    
    for sentence in sentences:
        #print(sentences)
        if "technologist receipt of the results" in sentence:
            continue
        elif ("by dr." in sentence) or ("dr." in sentence):
            #print("yes")
            continue
        elif ("telephone at xx p.m" in sentence) or ("telephone" in sentence) or ("p.m" in sentence) or ("a.m" in sentence):
            continue
        elif ("were discussed" in sentence) or ("department at" in sentence) or ("was called" in sentence) and ("informed of these critical results" in sentence) or ("was called informed of these critical results" in sentence):
            continue
        else:
            cleaned_text.append(sentence)
            
 
    cleaned_text = " . ".join(cleaned_text)
    #cleaned_text = re.sub(".\s+.", ".",cleaned_text)
    cleaned_text = re.sub("\s{2,10}"," ", cleaned_text)
    
    return cleaned_text


def preprocess():
    with open("./data/subset_images_captions.json", "r") as f:
        images_captions = json.load(f)
    
    if os.path.exists("./data/data_shard/cleaned_captions_index.json"):
        with open("./data/data_shard/cleaned_captions_index.json","r") as f:
            temp = json.load(f)
            cleaned_caption_indices = []
            for num in temp:
                cleaned_caption_indices.append(num)
    else:
        cleaned_caption_indices = []
        
    if os.path.exists("./data/data_shard/subset_cleaned_images_captions.json"):
        with open("./data/data_shard/subset_cleaned_images_captions.json", "r") as f:
            cleaned_image_captions = json.load(f)
    else:
        cleaned_image_captions  = []
        
    if os.path.exists("./data/data_shard/report_cache.json"):
        with open("./data/data_shard/report_cache.json", "r") as f:
            report_cache = json.load(f)
    else:
        report_cache  = {}
        
    if os.path.exists("./data/data_shard/indication_cache.json"):
        with open("./data/data_shard/indication_cache.json", "r") as f:
            indication_cache = json.load(f)
    else:
        indication_cache  = {}
        
    if len(cleaned_caption_indices) == 0:
        count = 0
    else:
        count = cleaned_caption_indices[-1]+1
    #count = 250 #304
    #problem_cache = {}
    images_captions = images_captions[count:]
    
    question_template = list(set(QUESTION_TEMPLATE))
    
    # for i in range(25):
    #     question_template = random.sample(question_template, len(question_template))
    
        
    for ind, each in enumerate(images_captions):
        # if ind in cleaned_caption_indices:
        #     continue
        
        print(f"Processing sample {count+ ind}...")
        extra_samples = []
        sample = {}
        sample["image"] = each["image"]
        sample["type"] = "original"
        
        problems = each["problems"]
        caption = each['caption']
        
        if "xx" in caption:    
            if caption not in report_cache.keys():
                #Gpt3.5 here
                print("Cleaning caption...")
                caption = chat_openai(report=caption,template=1).replace("[","").replace(']','').replace("Chest:","")
                caption = "\n".join(caption.split("\n\n")[:3])
                #TODO preprocess results
                
                sample["caption"] = caption
                previously_unclean = True
            else:
                sample["caption"] = report_cache[each["caption"]]
                caption = report_cache[each["caption"]]
                previously_unclean = False
        else:
            sample["caption"] = caption
            previously_unclean = False
                
        sample["problems"] = problems
        # GPT3.5 here
        indication = each["indication"]
        #print("Cleaning indication...")
        if indication not in indication_cache.keys():
            indication = chat_openai(text= indication.replace("r/o","rule out").replace('tb','tuberculosis'), template=3)
            indication = re.sub(".\s*No indication","", indication)
            indication_cache[each["indication"]] = indication
        else:
            indication = indication_cache[indication]  
            
        with open("./data/data_shard/report_cache.json","w") as f:
            json.dump(report_cache,f)
            
        with open("./data/data_shard/indication_cache.json","w") as f:
            json.dump(indication_cache,f)    
        
        # Get extra samples using question answering
        question_result = []
        if caption not in report_cache.values():
            question_result.extend(get_qa(each["image"], caption, problems, indication, question_template))
        
        temp_indication =  "<ind> " + indication + " <ind>"
        sample["indication"] = add_prompt(temp_indication)
        extra_samples.append(sample)
        #break  #678
        if len(question_result) > 0:
            extra_samples.extend(question_result) 
                   
        cleaned_image_captions.extend(extra_samples)
        
        if previously_unclean:
            report_cache[each["caption"]] = caption
            
        with open("./data/data_shard/subset_cleaned_images_captions.json","w") as f:
            json.dump(cleaned_image_captions,f)
            
        cleaned_caption_indices.append(count +ind)
        with open("./data/data_shard/cleaned_captions_index.json","w") as f:
            json.dump(cleaned_caption_indices, f)
            
        
        #count += 1
        # if count == 1:
        #     break
        
        
    # with open("./data/data_shard/cleaned_images_captions.json","w") as f:
    #     json.dump(cleaned_image_captions,f)
    
    df = pd.DataFrame(cleaned_image_captions)
    df.to_csv("./data/data_shard/subset_cleaned_images_captions.csv")
    
    return cleaned_image_captions

def remove_punc(string, remove_prob=0.5):
    keep_punc = random.choices([0,1],weights = [remove_prob, 1- remove_prob])[0]
    if keep_punc:
        return string
    else:
        string = string.replace("?", "")
        return string

def add_prompt(string, add_prob=0.5):
    templates = ["Give a detailed report of the image.",
                 'Explain this image',
                 'Explain the image given',
                 "Describe this image to me.",
                 "Can you give a detailed explanation of this x-ray image?"]
    add_ppt = random.choices([0,1],weights = [add_prob, 1- add_prob])[0]
    if add_ppt:
        ppt = random.sample(templates, 1)[0]
        string = string + "<prompt> " + ppt + "<prompt>"
        return string
    else:
        return string
    
def rm_indication(string, rm_prob=0.4):
    rm_ind = random.choices([0,1],weights = [1 - rm_prob, rm_prob])[0]
    if rm_ind:
        string = string.split("<ind>")[-1]
        return string
    else:
        return string

def add_noise(string, add_noise_prob=0.2, noise_prob=0.05 ):
  insert_noise = random.choices([0,1], weights=[1 - add_noise_prob, add_noise_prob])[0]
  if not insert_noise:
    return string
  else:
    # Add noise : insert new character or remove character:
    remove_character = random.choice([0,1])
    if remove_character:
      n_characters = round(noise_prob * len(string))
      char_indices = random.sample(list(range(len(string))), n_characters )
      edited_string = ""
      for pos, i in enumerate(string):
        if pos not in char_indices:
          edited_string += i
      #return edited_string
    else:
      n_characters = round(noise_prob * len(string))
      new_characters = random.sample(characters, n_characters)
      positions = random.sample(range(len(string)), n_characters)
      edited_string = list(string)
      for c , p in zip(new_characters, positions):
        edited_string.insert(p, c)
      edited_string  = "".join(edited_string)

    return edited_string

    

def get_qa(image_name, report, problems, indication, question_template):
   #mention of,     
    data = []
    question_list = random.sample(question_template, 30)
    
    print("Fetching answers to all questions...")
    
    answers = chat_openai(report= report, text= "".join(question_list),template=2).split("\n")
        # print(f"Length of answers for {organ}: " ,len(answers))
        # print(f"Length of questions for {organ}: ", len(question_list))
        # print(question_list)
        # print(answers)
        # print("------------------------------")
    print(len(answers), len(question_list))
    
    # for i, q in enumerate(question_list):
    #     print( q, "\n", answers[i], "\n") 
        
    # if len(answers) != len(question_list):
    #     answers = answers[1:]
    
    # if len(answers) != len(question_list):
    #     answers = answers[1:]
        
    if len(answers) != len(question_list):
        print(f"Number of questions : {len(question_list)} != Number of answers : {len(answers)}.\nReturning empty list...")
        return data
    # Better results forom 858
    for num, each_question in enumerate(question_list):
        each_question = remove_punc(re.sub("\d+.\s*", "",each_question))
        new_sample = {}
        new_sample["image"] = image_name
        new_sample["indication"] = "<ind>" + indication + "<ind>" + "<prompt> " + each_question + " <prompt>"
        new_sample["problems"] = problems
        new_sample["type"] = "question_answering"
        answer  = answers[num]
        #the report does not mention any, The report does not provide information ,the report confirms (the)
        start_index = re.search("\d+.",answer).span()[1]
        answer = answer[start_index:]
        #print(answer)
        answer = answer.replace('mentioned in the report',"seen in the image").replace("if applicable","").replace("are reported as","are seen as")
        answer = answer.replace("[","").replace("]","").replace('The report does not mention any abnormalities', 'There are no abnormalities').replace("The report does not mention any","There is no evidence of").replace("The report suggests", "There is")
        answer = answer.replace('There is no mention of', "There is no").replace("The report mentions","There is").replace("The report confirms","There is")
        answer = answer.replace("The report","The image").replace("the report","the image").replace("report", "image").replace("Report","Image").replace("mentioned","seen").replace("mentions","notes")
        
        new_sample["caption"] =  answer# might add notable findings here.
        data.append(new_sample)
        
    return data

def chat_openai(report=None, text=None, template=1, temperature =0.):
    if template == 1:
        system_message = return_report_template(report)
    elif template ==2:
        system_message  = return_question_template(report,text)
    elif template ==3:
        system_message = return_indication_template(text)
    else:
        system_message = return_problems_template(text)
            
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            #{"role": "system", "content": system_message},
            {"role": "user", "content": system_message},
            
        ],
    temperature=temperature
    )
    response_message = response['choices'][0]['message']['content']
    return response_message


text = """"findings: the heart is normal in size . the mediastinum is stable . left-sided chest xx is again visualized with tip at \
        cavoatrial junction . there is no pneumothorax . numerous bilateral pulmonary nodules have increased in size and number xx . \
            the dominant nodule/mass in the left midlung is also mildly increased\nproblems: catheters; indwelling; nodule; nodule\n\
            impression: there is no pleural effusion . 
            """
# reports = response.split("\n\n")[:3] are reported as

question_list = [
    "Can you confirm the absence of diaphragmatic rupture or diaphragmatic eventration?",
    "Do you notice any signs of diaphragmatic hernias, such as hiatal hernia?",
    "Are there any signs of diaphragmatic tumors, such as leiomyoma or lipoma?",
    "Can you rule out any signs of diaphragmatic lymphadenopathy or metastases?",
    "Are both hemidiaphragms moving symmetrically during respiration?",
    "Do you notice any signs of phrenic nerve injury or paralysis?",    
    "Can you confirm the absence of diaphragmatic eventration or diaphragmatic thinning?",
    "Do you notice any signs of diaphragmatic paralysis on fluoroscopy?",
    "Can you rule out any signs of Morgagni hernia or Bochdalek hernia?",
    "Are there any signs of diaphragmatic rupture or diaphragmatic tear?",
    "Is the diaphragm adequately positioned without any signs of upward or downward displacement?",
    "Do you notice any signs of diaphragmatic lymphangioma or diaphragmatic lipoma?",
    "Can you rule out any signs of diaphragmatic metastases or diaphragmatic abscess?",
    "Are both hemidiaphragms moving symmetrically during respiration?",
    "Do you notice any signs of phrenic nerve injury or phrenic nerve palsy?",
    ]

indication = "-year-old male for preop evaluation."

# report = 0.75, questions=0, indication = 0., problems=0.
if __name__ == '__main__':
    # df = pd.read_csv('./data/image_captions.csv') #create_df()
    # clean_text(df)
    text = """"findings: the heart is normal in size . the mediastinum is stable . left-sided chest xx is again visualized with tip at \
        cavoatrial junction . there is no pneumothorax . numerous bilateral pulmonary nodules have increased in size and number xx . \
            the dominant nodule/mass in the left midlung is also mildly increased\nproblems: catheters; indwelling; nodule; nodule\n\
            impression: there is no pleural effusion . 
            """
    
    #clean_convert_numbers()
    #create_df()
    preprocess()
    # response = chat_openai(report= text,template=1)
    # print(response)
    # print("\n".join(response.split("\n")[:3]))
    

    # TODO
    #two, multiple
    # 9th, fourth
    #T9,T6, T8, T11 and T12
    
    
    
    #After GPT 3.5
    #convert decimal to integers
    #Remove 'Chest:'
    #[chest wall?], [adjacent area?], 
    #preprocess()

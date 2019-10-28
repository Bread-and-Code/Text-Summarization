def summary_cleaner(text):
    newString = re.sub('"','', text)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = newString.lower()
    tokens=newString.split()
    newString=''

    for i in tokens:
        if len(i)>1:
            newString=newString+i+' '
    return newString

 #Call the above function

 cleaned_summary = []
 for t in data['summary']:
     cleaned_summary.append(summary_cleaner(t))


 data['cleaned_text']=cleaned_text
 data['cleaned_summary']=cleaned_summary
 data['cleaned_summary'].replace('', np.nan, inplace=True)
 data.dropna(axis=0,inplace=True)

import matplotlib.pyplot as plt
text_word_count = []
summary_word_count = []

# populate the list with the sentence lengths

for i in data['cleaned_text']:
    text_word_count.append(len(i.split()))

for i in data['cleaned_summary']:
    summary_word_count.append(len(i.split()))


length_df = pd.Dataframe({'text':text_word_count,'summary':summary_word_count})
length_df.hist(bins = 30)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

def get_most_important_features(vectorizer, model, n=5):
    """
    This function returns the most important words
    starting from the most important to he least important 
    that the algorithm used to learn
    Inputs:
        vectorizer >> The Vectorizer object TF-Idf or CountVectorizer
        model >> the algorithm object of LogisticRegression class
        n >> the number of returned words
    """
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    """
    This function plots the important words
    according to its scores    
    """
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    
    fig = plt.figure(figsize=(10, 10))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Objective', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Subjective', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplots_adjust(wspace=0.8)
    plt.show()
    
    
    
#importance = get_most_important_features(vectorizer, log_reg, 10)
# print (importance)
#top_scores = [a[0] for a in importance[0]['tops']]
#top_words = [a[1] for a in importance[0]['tops']]
#bottom_scores = [a[0] for a in importance[0]['bottom']]
#bottom_words = [a[1] for a in importance[0]['bottom']]

#plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")

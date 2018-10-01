def get_headline(headlines_df):
    
    # get headlines as list
    headlines_lst = []
    for row in range(0,len(headlines_df.index)):
        headlines_lst.append(headlines_df.iloc[row])

    # split headlines to separate words
    basicvectorizer = CountVectorizer()
    headlines_vectorized = basicvectorizer.fit_transform(headlines_lst)
    
    print(headlines_vectorized.shape)
    return headlines_vectorized, basicvectorizer

def headline_mapping(target, headlines_vectored, headline_vectorizer):
    
    # round target values if using logistic regression
    target = round(target,0)
    
    # get model (testing with model that isn't )
    from sklearn.naive_bayes import MultinomialNB
    headline_model = MultinomialNB()
    headline_model = headline_model.fit(headlines_vectored, target)
    
    return headline_vectorizer, headline_model
    
#     # get coefficients
#     basicwords = headline_vectorizer.get_feature_names()
#     basiccoeffs = headline_model.coef_.tolist()[0]
#     coeff_df = pd.DataFrame({'Word' : basicwords, 
#                             'Coefficient' : basiccoeffs})
    
#     # convert dataframe to dictionary of coefficients
#     coefficient_map = dict(zip(coeff_df.Word, coeff_df.Coefficient))
    
# #     if 'worldwide' in coefficient_map:
# #         print('yes') 
    
#     return coefficient_map, coeff_df['Coefficient'].mean()

def headlines_predict(headlines_df, headline_vectorizer, headline_model):
    
    # get headlines as list
    headlines_lst = []
    for row in range(0,len(headlines_df.index)):
        headlines_lst.append(headlines_df.iloc[row])
        
    # apply vectorizer
    headlines_test = headline_vectorizer.transform(headlines_lst)
    return headline_model.predict(headlines_test)

headline_vectorizer, headline_model = headline_mapping(X_train['returnsOpenNextMktres10'],
                                            *get_headline(X_train['headline']))

predictions = headlines_predict(X_train['headline'], headline_vectorizer, headline_model)

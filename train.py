if __name__=="__main__":
    from scripts.DeepModel import x_train, x_test, y_train, y_test
    from scripts.DeepModel import large_cnn_model, batch_size
    
    print(large_cnn_model.summary())
    large_cnn_model.fit(x_train, y_train, 
          validation_data=(x_test, y_test), 
          epochs=10, batch_size=200, verbose=2)
    
    large_cnn_scores = large_cnn_model.evaluate(x_test, y_test, verbose=0)
    print("Large CNN Accuracy: {:.2f}%".format(large_cnn_scores[1]*100))
    print("Large CNN Error: {:.2f}%".format(100-large_cnn_scores[1]*100))
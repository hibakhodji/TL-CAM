import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
class QXP:
    
            
   
    def produce_plot(self, weights, max_preds, unique_max_preds, labels, lim):
        #print(unique_max_preds)
        #print(labels)
        
        labels_plot = [self.labels[i] for i in unique_max_preds]
        #print(labels_plot)
        y_pos = np.arange(len(labels_plot))
        width = 0.2
        
            

        # Contribution by score on target class                
        scores_for_plot = [float(w) for w in weights]
        label_score = {k: v for k, v in zip(labels_plot, scores_for_plot)}
        label_score_sorted = dict(sorted(label_score.items(), key=lambda item: item[1], reverse=True))

      
         # Contribution by number of feature maps in last conv layer
        maxpreds = [max_preds.count(i) for i in unique_max_preds]
        #print('labels_plot', labels_plot)
        #print('----------------------------------')
        #print('maxpreds', maxpreds)
        #print('----------------------------------')


        label_max = {k: v for k, v in zip(labels_plot, maxpreds)}
        label_max_sorted = {k: label_max[k] for k in label_score_sorted}
        
        
        
        #print("----------------------------- maxpreds not sorted", label_max)
        #print("----------------------------- maxpreds sorted", label_max_sorted)
        
        
        
        info=list(label_score_sorted.keys())

        fig = go.Figure(
            data=[
                go.Bar(name='Number of feature maps', x=info, y=list(label_max_sorted.values()), yaxis='y',
                       text=[round((i*100))/lim for i in list(label_max_sorted.values())],
                       texttemplate = "%{text:.1f}%",
                       textposition='auto', offsetgroup=1, textfont_size=14),
                go.Bar(name='Average score', x=info, y=list(label_score_sorted.values()), yaxis='y2',
                       texttemplate = "%{y:.2f}",
                       textposition='outside', offsetgroup=2, textfont_size=14)
                ],
            layout={
                    'yaxis': {'title': 'Number of feature maps', 'dtick':'20'}, 
                    'yaxis2': {'title': 'Average prediction score on target class', 'overlaying': 'y', 'side': 'right'}
                }
            )

        # Change the bar mode
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='group',
        bargap=0.30,
        bargroupgap=0.0,
        width=800,
        height=600,
        yaxis=dict(range=[0,lim]))
        fig.show()
       
    def __init__(self, score=None, predictions=None, labels=None, shape=None, **kwargs):
        super().__init__(**kwargs)
        self.score = score
        self.predictions = predictions
        self.labels = labels
        self.shape = shape

    def explain(self):
        
        
        #print("score QXP", self.score)
        
        max_preds = tf.argmax(self.predictions, axis=-1)
        #print("max_preds tensor", max_preds)
        #print('------------------------------------')
        unique_max_preds = set(max_preds.numpy())
        #print("unique_max_preds set", unique_max_preds)
        #print('------------------------------------')
        
        
        # Target weights per feature map
        feature_map_target_weights = tf.gather(self.predictions, self.score, axis=-1) 
        
        #print("feature_map_target_weights", feature_map_target_weights)
        _indices_per_class = [tf.reshape(tf.where(tf.math.equal(tf.argmax(self.predictions, axis=-1), unique_max_pred)), (-1)) for unique_max_pred in unique_max_preds]
        
        #for i in _indices_per_class: print("_indices_per_class", i)
            
        weights_per_target_class = [tf.gather(feature_map_target_weights, indices, axis=0) for indices in _indices_per_class]            
        
        #print('------------------------------------')
        
        #for w in  weights_per_target_class: print("weights_per_target_class", w)
        weights = [tf.math.reduce_mean(w) for w in weights_per_target_class]
        
        #print('------------------------------------')
        #for w in weights: print("weights", w)
            
        # Producing plot
        
        self.produce_plot(weights, tf.unstack(max_preds), unique_max_preds, self.labels, self.shape)

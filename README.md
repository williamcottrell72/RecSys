# RecSys

This repo will contain some prototypes for recommender systems.  

At present, the primary recommender system being used is contained in recsys.py.  The file 'recommender.py' calls recsys.py, feeds it some data scraped from trip adviser and then combines the neural-network based recommendations of recsys.py with a more traditional recommender.  

The recommender.py file is specific to trip adviser, while recsys.py could be applied to any recommendation problem.  

At present, the recommender.py may be used as follows:

```python
import recommender
preferences= [x1,x2,x2,x4]
recommender.preferences_to_placescores(preferences,num_results=20,weight=.01)
````

## EASDRL
## Code for the IJCAI-18 paper 'Extracting Action Sequences from Texts Based on Deep Reinforcement Learning'

### requriements:

> tensorflow / keras
>
> wxpython
>
> gensim
>
> ipdb 
>
> ...



* PS: There should be a folder named 'weights', but it was automatically removed by github since it is empty. Remember to add it if you want to train a new model.

* ```bash
  mkdir weights
  ```



## Running

**Training:**  All arguments are preset in main.py, so you can start training by:  

```bash
$ python main.py
```

For trianing Argument Extractor, you can run:

```bash
$ python main.py --agent_mode arg
```

If you want to change the domain from 'cooking' to 'win2k' or 'wikihow', try:

```bash
$ python main.py --domain win2k
```

It may takes 2-4 hours for 'win2k', 10-15 hours for 'cooking' and 20-30 hours for 'wikihow' in our computer with TITAN Xp GPU. Change the size of replay memory, GPU fraction or number of epochs according to your servers.



**Human-agent Interaction:** If you want to use the interacting environment,   make sure you have installed the wxpython, and try:

```bash
$ python gui.py
```

It's the initial version, which is simple and maybe has some bugs. We will update it in the near future.  



This repository depicts the current state of my undergraduate thesis. (work ongoing)

    ├── /clustering           # core module
    │   ├── /em               ## low-level implementation of the EM-Algorithm
    │   ├── /initialization   ## starting value routines (multiple options)
    │   ├── /input_data       ## simulate/load experimental data
    │   ├── /miscellanous     ## uncategorized tools
    │   ├── /model_selection  ## select best models according to specific criteria
    │   └── clustering.py     ## high-level access point for application
    └── ...

**Installation** (currently not recommended): 
* clone repository
* set up a new virtual environment
* obtain required packages via ```pip install -r path/to/requirements.txt``` 
 
**Tutorial**:  
  
Feel free to have a look at ```example_notebook_README.ipynb``` if you want to see how the example in the *Project Overview* below has been generated. 
    
## Project Overview
### Objective

The purpose of this project is to create a fully automated unsupervised learning algorithm to identify statistical parameters of individual defects from measurements made by time-dependent defect spectroscopy (TDDS) of MOSFETs.
  
*Starting with the input data in the left picture; the goal is to assign each point to its generating defect. The true assignments can be observed in the right picture as colored clusters.*
![cluster_figure](https://user-images.githubusercontent.com/97874941/207207841-bc978c52-2cd5-4f18-b1fe-e30661fea504.svg)

### Idea

The reasonable assumption, that the measurements can be modelled 
by a bivariate mixture distribution, incentivizes the use of a model-based clustering approach. After promising initial results 
the decision was made to utilize the Expectation-Maximization Algorithm (EM-Algorithm).

### Problems (to be) solved
Although the EM-Algorithm has proven itself over a wide range of application areas in the past, 
it possesses some inconveniences, which make it hard to fully automate the clustering process:

* the outcome is highly dependent on starting values
* it cannot decide on the correct number of clusters   
* outliers can distort the outcome
* isolated datapoints can lead to useless solutions in the form of mixtures with divergent likelihoods

 
### Flowchart
This flowchart gives you a rough idea on how the module works and how its different components relate to each other.  
  
(reload page if the chart does not render)
```mermaid
flowchart TD
    A(/input_data) -- load experimental data --> J(clustering.py)
    J -- parallelized --> B(/initialization)
    B --> H{{for different clusternumbers\nand multiple runs}}
    H -- starting values --> C(/em)
    B-. new starting values .-> C
    C --> G{misbehaving\nresult?}
    G-. Yes .-> B
    G -- No --> D(/model_selection)
    D --> E(((best model)))
    
    style J stroke-width:6px
    style B stroke-width:4px,stroke-dasharray: 5 5
    style A stroke-width:4px,stroke-dasharray: 5 5
    style D stroke-width:4px,stroke-dasharray: 5 5
    style C stroke-width:4px,stroke-dasharray: 5 5
```
    
  
## Interim results
This is the result for the simulated input data from the *Objective* section above clustered by this module. The figure can be used to analyse the outcome of the clustering in detail.
 
  

*We can see that the algorithm sucessfully identified cluster 0 (red) and 4 (lime green), the assignments of the points are marked with the letters c and b, respectively. Cluster 2 (dark green) and 5 (pink) have been merged together with their bigger neighbours 1 (violet) and 3 (teal).*
![result_best_model](https://user-images.githubusercontent.com/97874941/207207821-f3f879fa-a809-4528-8470-09d96e48fa87.svg)


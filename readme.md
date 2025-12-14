## Recommendation system using graph network
we are try to build a system of recommendation by using SNA twitter dataset, where user have follwing, interests(#), features(@).
Objective is to make a recommendation system with solving the fold start problem.

### structure of the file
`
userid -->  1.circles (user id)
            2.edges ( connection of user)
            3.egofeat (user interests mapping)
            4.feat (user interest)
            5.featnames (feature names)
`
### structure of the graph

    Nodes --> (user, interest, follwing)
    edges --> connection between them

## how to run the project

#### step 1: create a virtual environment & activate it by following commands
`
    python -m venv myenv
`
`
    myenv\Scripts\activate
`
#### step 2: install all reuired library
`
    pip install -r Recommender app\requirements.txt
`

#### step 3: Run the app
`
    streamlit run Recommender app\app.py
`
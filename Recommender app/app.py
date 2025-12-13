import streamlit as st
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import os
from recommender import TwitterRecommender 

# 1. APP CONFIG
st.set_page_config(page_title="FriendLink Recommendation", layout="wide")

# Hide Streamlit default menu for a cleaner look
st.markdown("""
    <style>
    .block-container {padding-top: 1rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("FriendLink: Static Graph Analysis")

# 2. CACHED DATA LOADER
@st.cache_resource
def load_data():
    graph_path = 'models/twitter_graph.gpickle'
    model_path = 'models/twitter_node2vec.model'
    
    if not os.path.exists(graph_path):
        return None, None
        
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
        
    rec_engine = TwitterRecommender(model_path)
    return G, rec_engine

G, rec_engine = load_data()

if G is None:
    st.error("❌ Data not found. Please run 'Linkprediction.ipynb' then 'train_model.py'.")
    st.stop()

# 3. SIDEBAR CONTROLS
st.sidebar.title("Controls")
mode = st.sidebar.radio("Mode", ["Existing user Simulator", "New User Simulator"])
max_neighbors = st.sidebar.slider("Max Nodes to Show", 10, 100, 30, 
                                  help="Limit the number of nodes to keep rendering fast.")

# 4. MAIN LOGIC
if mode == "Existing user Simulator":
    # Helper to get user list efficiently
    @st.cache_data
    def get_user_list(_graph):
        return [n for n, d in _graph.nodes(data=True) if d.get('type') == 'user']

    all_users = get_user_list(G)
    
    if not all_users:
        st.error("No users found in the graph. Check your data processor.")
        st.stop()

    selected_user = st.sidebar.selectbox("Select Target User", all_users[1:200])

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"Ego Network Snapshot: {selected_user}")
        
        # --- SUBGRAPH LOGIC ---
        try:
            all_neighbors = list(G.neighbors(selected_user))
        except:
            all_neighbors = []
            
        # Limit neighbors for speed
        if len(all_neighbors) > max_neighbors:
            neighbors = all_neighbors[:max_neighbors]
        else:
            neighbors = all_neighbors
        st.caption(f"⚠️ User may have many connections. Showing top {len(neighbors)} to prevent complexity.")

        nodes_to_draw = [selected_user] + neighbors
        subgraph = G.subgraph(nodes_to_draw).copy()
        existing_edges = list(subgraph.edges)

        # Recommendation engine
        future_friends=[]
        if rec_engine:
            recs = rec_engine.find_friends_for_existing_user(selected_user, all_neighbors)
            if recs:
                for user, score in recs:
                    subgraph.add_node(user, type='future_prediction')
                    subgraph.add_edge(user, selected_user, type='predicted')
                    future_friends.append(user)

                    # for existing_node in nodes_to_draw:   
                    #     if G.has_edge(existing_node, user):
                    #         subgraph.add_edge(user, existing_node, type='Neighbours_connection')
        
        # --- STATIC MATPLOTLIB RENDERING ---
        if len(subgraph.nodes()) > 0:
            # 1. Calculate Positions (Spring Layout)
            # k controls spacing (higher = further apart)
            pos = nx.spring_layout(subgraph, k=0.6, iterations=50, seed=42)
            
            # Create Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 2. Filter Nodes by Type
            target_node = [selected_user]
            user_nodes = [n for n in neighbors if subgraph.nodes[n].get('type') == 'user']
            feat_nodes = [n for n in neighbors if subgraph.nodes[n].get('type') == 'feature']
            
            # 3. Draw Nodes (Distinct Styles)
            # Target User (Yellow, Big)
            nx.draw_networkx_nodes(subgraph, pos, nodelist=target_node, 
                                   node_color='#FFD700', node_size=800, label='Target', ax=ax)
            
            # Friend Users (Green, Medium)
            if user_nodes:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=user_nodes, 
                                       node_color='#71DF92', node_size=400, label='Friends', ax=ax)
            
            # Interests/Features (Red, Square)
            if feat_nodes:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=feat_nodes, 
                                       node_color='#FF6347', node_size=200, node_shape='s', label='Interests', ax=ax)
            
            # 4. Draw Edges (Gray, Thin)
            nx.draw_networkx_edges(subgraph, pos, edgelist=existing_edges, width=1.0, alpha=0.6, edge_color='grey', ax=ax)
            
            # 5. Draw Labels
            labels = {n: str(n).replace("Feat: ", "") for n in feat_nodes}
            user_labels = {n: n for n in user_nodes}
            labels[selected_user] = str(selected_user)
            
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold', ax=ax)
            nx.draw_networkx_labels(subgraph, pos, user_labels, font_size=5, ax=ax)
            
            ax.set_title("Before the recommendation")
            ax.axis('off') # Hide axis border
            st.pyplot(fig)

            # After the recommendation
            fig2, axis = plt.subplots(figsize=(10, 8))
            # Target User (Yellow, Big)
            nx.draw_networkx_nodes(subgraph, pos, nodelist=target_node, 
                                   node_color='#FFD700', node_size=800, label='Target', ax=axis)
            
            # Friend Users (Green, Medium)
            if user_nodes:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=user_nodes, 
                                       node_color='#71DF92', node_size=300, label='Friends', ax=axis)
            
            # Interests/Features (Red, Square)
            if feat_nodes:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=feat_nodes, 
                                       node_color='#FF6347', node_size=200, node_shape='s', label='Interests', ax=axis)
                
            # suggestions (Purple)
            if recs:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=future_friends, node_color='#D45F96', node_size=400, node_shape='o', label='recommedation', ax=axis)
            
            nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.6, edge_color='grey', ax=axis)
        
            future_friend_labels = {n: n for n in future_friends}
            future_friend_labels[selected_user] = str(selected_user)
            
            nx.draw_networkx_labels(subgraph, pos, future_friend_labels, font_size=8, font_weight='bold', ax=axis)
            
            axis.set_title("After the recommendation")
            axis.axis('off')
            st.pyplot(fig2)

    with col2:
        st.subheader("🤖 AI Analysis")
        st.write("Recommendations:")
        if recs:
            for user, score in recs:
                st.info(f"**User {user}**\n\nMatch: {score:.2f}")
        else:
            st.warning("No recommendations available.")

elif mode == "New User Simulator":
    st.subheader("Cold Start Simulation")
    
    @st.cache_data
    def get_features(_graph):
        return [n for n in _graph.nodes() if "#" in str(n)]

    all_features = get_features(G)
    
    if not all_features:
        st.error("No features/interests found in the graph.")
        st.stop()
        
    selected_interests = st.multiselect("Select User Interests", all_features[:101])
    
    if st.button("Suggest matching vibes"):
        if selected_interests:
            recs = rec_engine.cold_start_recommendation(selected_interests)
            st.success("Closest matches found based on Vector Centroid:")
            
            cols = st.columns(3, border=True)
            for i, (user, score) in enumerate(recs[:3]):
                with cols[i]:
                    st.metric("Friend Match User:", f"{user}", f"{score:.2f}")
        else:
            st.warning("Please select at least one interest.")
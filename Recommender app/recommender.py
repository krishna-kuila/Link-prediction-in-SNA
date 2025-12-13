from gensim.models import Word2Vec
import numpy as np

class TwitterRecommender:
    def __init__(self, model_path):
        # Load the pre-trained model
        self.model = Word2Vec.load(model_path)
        
    def find_friends_for_existing_user(self, user_id, all_neighbors, top_n=5):
        """
        Finds similar users based on graph structure.
        """
        if user_id not in self.model.wv:
            return []
            
        # Get most similar nodes
        similar_nodes = self.model.wv.most_similar(user_id, topn=top_n+50)
        
        # Filter: We only want to recommend Users, not Features (hashtags)
        recommendations = []
        for node, score in similar_nodes:
            if "Feat:" not in node and node != user_id and node not in all_neighbors:
                recommendations.append((node, score))
                if len(recommendations) >= top_n:
                    break
        return recommendations

    def cold_start_recommendation(self, interest_list, top_n=5):
        """
        The 'Vector Averaging' Strategy.
        interest_list: ['Feat: Tech', 'Feat: Politics']
        """
        valid_vectors = []
        for interest in interest_list:
            if interest in self.model.wv:
                valid_vectors.append(self.model.wv[interest])
        
        if not valid_vectors:
            return []

        # Calculate the "Centroid" (Average Vibe)
        user_vector = np.mean(valid_vectors, axis=0)
        
        # Find closest users to this synthetic vector
        similar_nodes = self.model.wv.similar_by_vector(user_vector, topn=top_n+10)
        
        recommendations = []
        for node, score in similar_nodes:
            if "Feat:" not in node: # Only recommend people
                recommendations.append((node, score))
                if len(recommendations) >= top_n:
                    break
        return recommendations
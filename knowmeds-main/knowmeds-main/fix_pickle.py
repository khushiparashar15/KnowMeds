# Load and verify the pickle file
with open('similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)

print("Pickle file loaded successfully!")
print("Similarity Matrix Shape:", similarity.shape)
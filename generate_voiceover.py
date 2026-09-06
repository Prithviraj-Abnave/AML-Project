from gtts import gTTS

script_segments = [
    "Welcome to the Comparative Topic Analysis dashboard. Here we benchmark Latent Dirichlet Allocation against Non-negative Matrix Factorization on 25,000 Amazon product reviews, examining coherence, diversity, and computational efficiency.",
    "Exploring the LDA Model tab: LDA treats each review as a probabilistic blend of topics. As we scroll down, you can see the top keyword clusters, word probability distributions, and dominant topic assignments.",
    "Switching to NMF: Using non-negative matrix factorization on the TF-IDF representation yields higher topic coherence (0.864) and near-instant execution, producing distinct, non-overlapping topic definitions.",
    "On the Compare tab, side-by-side metrics reveal NMF leads in coherence, diversity, and memory footprint. The normalized radar profile and similarity heatmap clearly map corresponding topics across both algorithms.",
    "Finally, running our pipeline on custom uploaded reviews. The dynamic model trainer builds interactive quality bar charts and radar profiles in real time, validating NMF's strong performance across custom text datasets."
]

full_script = " ".join(script_segments)

tts = gTTS(text=full_script, lang='en', slow=False)
tts.save("assets/voiceover.mp3")
print("Voiceover saved to assets/voiceover.mp3")

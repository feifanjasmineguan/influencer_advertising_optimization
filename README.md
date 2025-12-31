# influencer_advertising_optimization

Check out the live dashboard here: [Streamlit Dashboard Link](https://influenceradvertisingoptimization.streamlit.app)


This project studies the optimal selection of social media influencers for advertising campaigns under constraints using 400K node Twitter network data. We model the Twitter follower data as a directed graph and identify candidate influencers based on PageRank. To keep the computational costs at a reasonable level, a subset of users is randomly sampled to represent the target audience, and influencer candidates are restricted to those appearing within this sample. Since real demographic information is unavailable, we augment the network with synthetic user attributes, including age, gender, region, and primary interest, to simulate a realistic marketing context. The problem is formulated as an integer linear program and solved using Gurobi, with the objective of maximizing audience reach under constraints on budget, influencer reliability, demographic balance, frequency capping, and coverage.

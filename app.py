import pandas as pd
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import random
import os
import zipfile
import streamlit as st

# ------------------------------------------------------------------
# 1. APP CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Influencer Optimization Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Network-Based Influencer Optimization")
st.markdown("""
This tool uses **Linear Programming (Gurobi)** to select the optimal set of influencers 
based on budget, reliability, demographic targets, and network reach.
""")

# ------------------------------------------------------------------
# 2. GUROBI ENVIRONMENT SETUP (Cloud vs Local)
# ------------------------------------------------------------------
@st.cache_resource
def get_gurobi_env():
    """
    Initializes Gurobi Environment.
    Checks for Streamlit Cloud secrets first, then falls back to local license.
    """
    try:
        # Check if running on Streamlit Cloud with Secrets
        if "WLSACCESSID" in st.secrets:
            params = {
                "WLSACCESSID": st.secrets["WLSACCESSID"],
                "WLSSECRET": st.secrets["WLSSECRET"],
                "LICENSEID": int(st.secrets["LICENSEID"]),
            }
            return gp.Env(params=params)
        else:
            # Local machine: Uses gurobi.lic file
            return gp.Env()
    except Exception as e:
        st.warning(f"Using default Gurobi environment (License check failed or limited): {e}")
        return gp.Env()

env = get_gurobi_env()

# ------------------------------------------------------------------
# 3. DATA LOADING (Cached for Speed)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Loads graph/features directly from .zip files to save space/bandwidth.
    """
    # -------------------------------------------
    # 1. DEFINE PATHS (Assuming files are zipped)
    # -------------------------------------------
    # Ensure these match your actual filenames on GitHub
    nodes_zip_path = "all_nodes_features.csv.zip"
    edges_zip_path = "snowball_sample.edgelist.gz"
    
    # Internal filenames (what the file was named BEFORE zipping)
    # Usually the same as the zip name without '.zip'
    nodes_inner_name = "all_nodes_features.csv"
    edges_inner_name = "snowball_sample.edgelist"

    # Check existence
    if not os.path.exists(nodes_zip_path) or not os.path.exists(edges_zip_path):
        return None, None

    # -------------------------------------------
    # 2. READ NODES (Pandas handles zip automatically)
    # -------------------------------------------
    # Pandas is smart; if extension is .zip, it unzips automatically.
    nodes = pd.read_csv(nodes_zip_path)
    
    # Clean columns
    cols_drop = [c for c in nodes.columns if "Unnamed" in c]
    nodes.drop(columns=cols_drop, inplace=True, errors='ignore')

    # -------------------------------------------
    # 3. READ GRAPH (NetworkX needs help with .zip)
    # -------------------------------------------
    # We open the zip, find the file inside, and decode bytes to text
    with zipfile.ZipFile(edges_zip_path, "r") as z:
        with z.open(edges_inner_name) as f:
            # f is a byte stream. NetworkX needs strings.
            # We use a generator to decode line-by-line (memory efficient)
            lines = (line.decode("utf-8") for line in f)
            
            G = nx.read_edgelist(
                lines,
                create_using=nx.DiGraph(),
                nodetype=int
            )
    
    return nodes, G

with st.spinner("Loading Graph Data... (This may take a moment)"):
    nodes_df, G = load_data()

if nodes_df is None:
    st.error("❌ Data files not found! Please ensure `data/all_nodes_features.csv` and `data/higgs-social_network.edgelist` exist.")
    st.stop()
else:
    st.success(f"Data Loaded! Nodes: {len(nodes_df):,}, Edges: {G.number_of_edges():,}")

# ------------------------------------------------------------------
# 4. SIDEBAR INPUTS
# ------------------------------------------------------------------
st.sidebar.header("1. Optimization Constraints")

budget = st.sidebar.slider("Total Budget ($)", 1000, 50000, 5000, step=500)
min_reliability = st.sidebar.slider("Min Avg Reliability Score", 0.0, 1.0, 0.2, step=0.05)
female_perc = st.sidebar.slider("Min Female Audience (%)", 0, 100, 10, step=5) / 100.0
max_frequency = st.sidebar.slider("Max Frequency (Spam Cap)", 1, 50, 20)

st.sidebar.markdown("---")
st.sidebar.header("2. Simulation Size")
sample_size = st.sidebar.slider(
    "Audience Sample Size", 
    1000, 
    50000, 
    10000, 
    step=1000, 
    help="Higher numbers are more accurate but slower. Start small!"
)

# ------------------------------------------------------------------
# 5. MAIN LOGIC (Triggered by Button)
# ------------------------------------------------------------------
if st.button("🚀 Run Optimization", type="primary"):
    
    # --- A. Pre-Processing & Mapping ---
    with st.spinner("Sampling Graph & Building Maps..."):
        # Create mappings for fast lookup
        cost_map = nodes_df.set_index('node_id')['HiringCost'].to_dict()
        rel_map = nodes_df.set_index('node_id')["ReliabilityScore"].to_dict()
        gender_map = nodes_df.set_index('node_id')["gender"].to_dict()
        
        # Sampling
        all_node_ids = nodes_df["node_id"].tolist()
        # Safety check for sample size
        actual_sample_size = min(sample_size, len(all_node_ids))
        audiences = random.sample(all_node_ids, actual_sample_size)
        
        # Filter Candidates: Must be 'is_influencer_top1' AND in our sampled graph subset
        # (Assuming you want candidates to be part of the sampled graph context)
        # Note: If candidates are separate from audience, remove .isin(audiences) check
        candidates = nodes_df[
            (nodes_df["is_influencer_top1"] == 1) & 
            (nodes_df["node_id"].isin(audiences))
        ]["node_id"].tolist()

        if not candidates:
            st.error("No candidates found in the current sample! Try increasing sample size.")
            st.stop()

        # Build User -> Influencer Map (Inverted Index)
        user_to_influencer_map = {u: [] for u in audiences}
        
        # Optimization: Only iterate candidates to find their followers
        for cand in candidates:
            if G.has_node(cand):
                # Predecessors = Followers in a Directed Graph (u -> v means u follows v)
                followers = G.predecessors(cand)
                for follower in followers:
                    if follower in user_to_influencer_map:
                        user_to_influencer_map[follower].append(cand)

    # --- B. Gurobi Optimization ---
    with st.spinner("Solving Optimization Problem (Gurobi)..."):
        try:
            m = gp.Model("Streamlit_Marketing", env=env)
            m.setParam('OutputFlag', 0)  # Silence console
            m.setParam('TimeLimit', 60)
            m.setParam('MIPGap', 0.05)

            # Variables
            x = m.addVars(candidates, vtype=GRB.BINARY, name="x") # Hire
            y = m.addVars(audiences, vtype=GRB.BINARY, name="y")  # Reach

            # Objective: Maximize Reach
            m.setObjective(gp.quicksum(y[u] for u in audiences), GRB.MAXIMIZE)

            # Constraint A: Budget
            m.addConstr(
                gp.quicksum(x[i] * cost_map[i] for i in candidates) <= budget, 
                "Budget"
            )

            # Constraint B: Reliability
            # Sum(x * score) >= Threshold * Sum(x)
            m.addConstr(
                gp.quicksum(x[i] * rel_map.get(i,0) for i in candidates) >= 
                min_reliability * gp.quicksum(x[i] for i in candidates),
                "Reliability"
            )

            # Constraint C: Demographics
            # 1 if Female, 0 if Male
            # (1 - P) if Female, (-P) if Male
            coeffs = {}
            for u in audiences:
                g_val = gender_map.get(u, 'male') # default to male if missing
                if g_val == 'female':
                    coeffs[u] = 1.0 - female_perc
                else:
                    coeffs[u] = -female_perc

            m.addConstr(
                gp.quicksum(y[u] * coeffs[u] for u in audiences) >= 0,
                "Demographics"
            )

            # Constraint D & E: Coverage & Spam
            for u in audiences:
                potential_infl = user_to_influencer_map.get(u, [])
                if potential_infl:
                    # Variable for "times hit"
                    times_hit = gp.quicksum(x[v] for v in potential_infl)
                    
                    # Coverage: y[u] <= sum(x)
                    m.addConstr(y[u] <= times_hit, name=f"Cover_{u}")
                    
                    # Spam: sum(x) <= max_freq
                    m.addConstr(times_hit <= max_frequency, name=f"Spam_{u}")
                else:
                    # Impossible to reach
                    m.addConstr(y[u] == 0)

            # Solve
            m.optimize()

            # --- C. Results Display ---
            if m.Status == GRB.OPTIMAL:
                
                # Extract Selected Influencers
                selected_ids = [i for i in candidates if x[i].X > 0.5]
                total_cost = sum(cost_map[i] for i in selected_ids)
                avg_rel_score = sum(rel_map[i] for i in selected_ids) / len(selected_ids) if selected_ids else 0
                
                # Top Metrics
                st.markdown("### 🏆 Optimization Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Unique Reach", f"{int(m.ObjVal):,}")
                col2.metric("Influencers Hired", f"{len(selected_ids)}")
                col3.metric("Total Cost", f"${total_cost:,.2f}")
                col4.metric("Avg Reliability", f"{avg_rel_score:.2f}")

                # Detailed Table
                if selected_ids:
                    st.subheader("Hired Influencer List")
                    results_df = pd.DataFrame({
                        "Node ID": selected_ids,
                        "Cost": [cost_map[i] for i in selected_ids],
                        "Reliability": [rel_map[i] for i in selected_ids],
                        "Gender": [gender_map.get(i, "Unknown") for i in selected_ids]
                    })
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download Button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Hiring Plan",
                        csv,
                        "hiring_plan.csv",
                        "text/csv"
                    )
            else:
                st.warning("⚠️ No optimal solution found. The constraints might be too strict (try increasing budget or lowering reliability).")
        
        except Exception as e:
            st.error(f"An error occurred during optimization: {e}")
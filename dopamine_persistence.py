import numpy as np

def calculate_persistence(DRD2: float, BDNF: float, epsilon: float = 0.0) -> float:
    """
    Calculate the Dopamine Persistence using the formula:
    Persistence = 0.42 * DRD2 + 0.31 * BDNF + ε

    Parameters:
    - DRD2 (float): Dopamine receptor D2 level
    - BDNF (float): Brain-Derived Neurotrophic Factor level
    - epsilon (float): Random noise or adjustment factor (default is 0.0)

    Returns:
    - float: Calculated persistence value
    """
    persistence = 0.42 * DRD2 + 0.31 * BDNF + epsilon
    return persistence

# Example usage
if __name__ == "__main__":
    # Example values for DRD2 and BDNF
    DRD2_level = 1.5
    BDNF_level = 2.3
    epsilon_value = np.random.normal(0, 0.1)  # Adding some noise

    persistence_value = calculate_persistence(DRD2_level, BDNF_level, epsilon_value)
    print(f"Calculated Persistence: {persistence_value}")
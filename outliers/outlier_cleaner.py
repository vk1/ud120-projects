#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    errors = abs(net_worths - predictions) # Calculate errors

    cleaned_length = int(len(ages) * 0.9) # Removing 10% of data with largest error

    smallest_error_indices = errors.flatten().argsort()[:cleaned_length] # Get indices of smallest error

    cleaned_data = zip(ages.flatten()[smallest_error_indices],
                       net_worths.flatten()[smallest_error_indices],
                       errors.flatten()[smallest_error_indices])

    
    return cleaned_data


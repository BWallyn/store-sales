# =================
# ==== IMPORTS ====
# =================

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from store_sales.pipelines.data_preprocessing.nodes import merge_transactions

# ===============
# ==== TESTS ====
# ===============

def test_basic_merge():
    """Test a basic successful merge operation."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
        'store_nbr': [1, 2, 1],
        'other_data': ['A', 'B', 'C']
    })
    df_transactions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
        'store_nbr': [1, 2, 1],
        'transactions': [10, 20, 15]
    })
    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
        'store_nbr': [1, 2, 1],
        'other_data': ['A', 'B', 'C'],
        'transactions': [10, 20, 15]
    })
    result_df = merge_transactions(df.copy(), df_transactions.copy())
    assert_frame_equal(result_df, expected_df)

def test_no_matching_transactions():
    """Test merge when some rows in df have no matching transactions."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'store_nbr': [1, 1, 2],
        'other_data': ['A', 'B', 'D']
    })
    df_transactions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': [1],
        'transactions': [10]
    })
    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'store_nbr': [1, 1, 2],
        'other_data': ['A', 'B', 'D'],
        'transactions': [10.0, np.nan, np.nan] # Expect NaN for non-matches
    })
    result_df = merge_transactions(df.copy(), df_transactions.copy())
    assert_frame_equal(result_df, expected_df)

def test_empty_df_transactions():
    """Test merge with an empty transactions DataFrame."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'store_nbr': [1, 1],
        'other_data': ['A', 'B']
    })
    df_transactions = pd.DataFrame(columns=['date', 'store_nbr', 'transactions'])
    # Ensure correct dtypes for empty df_transactions if pandas infers them differently
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])
    df_transactions['store_nbr'] = df_transactions['store_nbr'].astype('int64') # or appropriate int type
    df_transactions['transactions'] = df_transactions['transactions'].astype('float64') # so merge doesn't create object type if all are NaN

    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'store_nbr': [1, 1],
        'other_data': ['A', 'B'],
        'transactions': [np.nan, np.nan]
    })
    result_df = merge_transactions(df.copy(), df_transactions.copy())
    assert_frame_equal(result_df, expected_df)

def test_empty_df():
    """Test merge with an empty primary DataFrame."""
    df = pd.DataFrame(columns=['date', 'store_nbr', 'other_data'])
    # Ensure correct dtypes for empty df
    df['date'] = pd.to_datetime(df['date'])
    df['store_nbr'] = df['store_nbr'].astype('int64')
    df['other_data'] = df['other_data'].astype('object')


    df_transactions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': [1],
        'transactions': [10]
    })
    expected_df = pd.DataFrame(columns=['date', 'store_nbr', 'other_data', 'transactions'])
    # Ensure correct dtypes for expected_df
    expected_df['date'] = pd.to_datetime(expected_df['date'])
    expected_df['store_nbr'] = expected_df['store_nbr'].astype('int64')
    expected_df['other_data'] = expected_df['other_data'].astype('object')
    expected_df['transactions'] = expected_df['transactions'].astype('float64') # Pandas default for merge with empty left and non-empty right numerical

    result_df = merge_transactions(df.copy(), df_transactions.copy())
    # For empty dataframes, dtypes can sometimes be tricky.
    # `check_dtype=False` can be used if exact dtype matching for empty columns is not critical,
    # but it's better to align them as above.
    assert_frame_equal(result_df, expected_df, check_dtype=False)


def test_transactions_with_extra_columns():
    """Test df_transactions having additional columns not in df."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': [1],
        'other_data': ['A']
    })
    df_transactions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01']),
        'store_nbr': [1, 2],
        'transactions': [10, 20],
        'payment_type': ['card', 'cash']
    })
    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': [1],
        'other_data': ['A'],
        'transactions': [10],
        'payment_type': ['card']
    })
    result_df = merge_transactions(df.copy(), df_transactions.copy())
    assert_frame_equal(result_df, expected_df)

def test_duplicate_keys_in_transactions():
    """Test behavior when df_transactions has duplicate keys (date, store_nbr).
       Pandas merge will create multiple rows in the result for each match from the left."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'store_nbr': [1, 1],
        'item_id': ['X1', 'X2']
    })
    df_transactions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01']),
        'store_nbr': [1, 1],
        'transactions': [10, 12] # Two transaction entries for the same date/store
    })
    # For '2023-01-01', store_nbr 1, df has 'X1'. It will be duplicated for each transaction.
    # For '2023-01-02', store_nbr 1, df has 'X2'. No match in transactions.
    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
        'store_nbr': [1, 1, 1],
        'item_id': ['X1', 'X1', 'X2'],
        'transactions': [10.0, 12.0, np.nan]
    })
    result_df = merge_transactions(df.copy(), df_transactions.copy())
    # Sort by 'transactions' to ensure order for comparison if merge order isn't guaranteed for duplicates
    # and item_id to make it fully deterministic.
    result_df = result_df.sort_values(by=['date', 'store_nbr', 'item_id', 'transactions']).reset_index(drop=True)
    expected_df = expected_df.sort_values(by=['date', 'store_nbr', 'item_id', 'transactions']).reset_index(drop=True)
    assert_frame_equal(result_df, expected_df)


# def test_key_column_type_mismatch_no_merge():
#     """Test merge fails for keys with incompatible types (e.g. int vs str)."""
#     df = pd.DataFrame({
#         'date': pd.to_datetime(['2023-01-01']),
#         'store_nbr': [1],  # int
#         'data': ['A']
#     })
#     df_transactions = pd.DataFrame({
#         'date': pd.to_datetime(['2023-01-01']),
#         'store_nbr': ['1'],  # string
#         'transactions': [100]
#     })
#     # Pandas merge will not match int(1) with str('1') on key columns
#     expected_df = pd.DataFrame({
#         'date': pd.to_datetime(['2023-01-01']),
#         'store_nbr': [1],
#         'data': ['A'],
#         'transactions': [np.nan] # Expect NaN as string '1' won't match int 1
#     })
#     result_df = merge_transactions(df.copy(), df_transactions.copy())
#     assert_frame_equal(result_df, expected_df)

def test_key_column_type_aligned_for_merge():
    """Test merge works if key types are aligned (e.g. after explicit conversion)."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': [1],
        'data': ['A']
    })
    df_transactions_aligned = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': ['1'], # Still string here
        'transactions': [100]
    })
    # Align the type before calling the function (simulate pre-processing)
    df_transactions_aligned['store_nbr'] = df_transactions_aligned['store_nbr'].astype(int)

    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': [1],
        'data': ['A'],
        'transactions': [100]
    })
    result_df = merge_transactions(df.copy(), df_transactions_aligned)
    assert_frame_equal(result_df, expected_df)

def test_date_as_string_compatible_format():
    """Test merge when 'date' columns are strings but in a compatible format."""
    df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'], # strings
        'store_nbr': [1, 1],
        'other_data': ['A', 'B']
    })
    df_transactions = pd.DataFrame({
        'date': ['2023-01-01'], # strings
        'store_nbr': [1],
        'transactions': [10]
    })
    # The merge will happen on the string representation.
    expected_df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'store_nbr': [1, 1],
        'other_data': ['A', 'B'],
        'transactions': [10.0, np.nan]
    })
    result_df = merge_transactions(df.copy(), df_transactions.copy())
    assert_frame_equal(result_df, expected_df)

def test_all_columns_present_after_merge():
    """Ensure original df columns and new df_transactions columns are present."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': [1],
        'data_col_df': ['val1']
    })
    df_transactions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_nbr': [1],
        'transactions': [50],
        'tx_info': ['info1']
    })
    result_df = merge_transactions(df.copy(), df_transactions.copy())
    expected_columns = ['date', 'store_nbr', 'data_col_df', 'transactions', 'tx_info']
    assert all(col in result_df.columns for col in expected_columns)
    assert len(result_df.columns) == len(expected_columns)

def test_merge_preserves_left_df_rows():
    """Explicitly check that all rows from the left DataFrame are preserved."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'store_nbr': [1, 2, 3],
        'unique_id': ['id1', 'id2', 'id3']
    })
    df_transactions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-04']),
        'store_nbr': [1, 3, 4], # Match for store 1 and 3, no match for 2, extra for 4
        'transactions': [10, 30, 40]
    })
    result_df = merge_transactions(df.copy(), df_transactions.copy())

    # Check if all original unique_ids are present (indirectly checks row preservation)
    assert len(result_df) == len(df) # Basic check for left merge without duplicates in right
    assert set(df['unique_id']) == set(result_df['unique_id'])

    # More robust check for row content preservation (excluding the newly merged column)
    # This ensures that the original part of the dataframe is untouched where there are no duplicate key expansions
    pd.testing.assert_frame_equal(
        result_df[['date', 'store_nbr', 'unique_id']].sort_values(by=['date', 'store_nbr']).reset_index(drop=True),
        df[['date', 'store_nbr', 'unique_id']].sort_values(by=['date', 'store_nbr']).reset_index(drop=True)
    )

def test_merge_with_pd_na_in_keys():
    """Test how merge handles pd.NA or np.nan in merge key columns.
    Pandas merge typically does not match NaN/NA keys with other NaN/NA keys."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', pd.NaT]), # NaT in date
        'store_nbr': [1, pd.NA, 3], # pd.NA in store_nbr (requires nullable int type)
        'other_data': ['A', 'B', 'C']
    }).astype({'store_nbr': 'Int64'}) # Use nullable integer type

    df_transactions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', pd.NaT]),
        'store_nbr': [1, pd.NA, 3],
        'transactions': [10, 20, 30]
    }).astype({'store_nbr': 'Int64'})

    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', pd.NaT]),
        'store_nbr': [1, pd.NA, 3],
        'other_data': ['A', 'B', 'C'],
        # np.nan/pd.NA in keys are not matched by pandas merge by default
        'transactions': [10.0, 20, 30]
    }).astype({'store_nbr': 'Int64', 'transactions': 'float64'})

    result_df = merge_transactions(df.copy(), df_transactions.copy())
    assert_frame_equal(result_df, expected_df, check_dtype=False)

    # Test with np.nan in a float key column (if store_nbr was float)
    df_float_key = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'store_nbr': [1.0, np.nan], # np.nan in store_nbr
        'other_data': ['A', 'B']
    })
    df_transactions_float_key = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'store_nbr': [1.0, np.nan],
        'transactions': [10, 20]
    })
    expected_float_key_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'store_nbr': [1.0, np.nan],
        'other_data': ['A', 'B'],
        'transactions': [10.0, 20.0] # np.nan keys don't match
    })
    result_float_key_df = merge_transactions(df_float_key.copy(), df_transactions_float_key.copy())
    assert_frame_equal(result_float_key_df, expected_float_key_df, check_dtype=False)


"""
This module provides the GeolocationProcessor class, which offers methods for processing and enriching
fraud detection datasets with geolocation information based on IP addresses.

Main functionalities include:
- Converting IP addresses to their integer representations for efficient comparison and merging.
- Merging transaction data with IP-to-country mapping data to append country information to each transaction.
- Utility methods to find the corresponding country for a given IP address using IP range lookups.

These utilities are designed to support geolocation-based risk analysis and feature engineering in fraud detection workflows.
"""


import ipaddress
import numpy as np

class GeolocationProcessor:
    # The following function converts a pandas Series of IP addresses to their integer representations.
    def ip_to_int(self, ip_series):
        """Convert IP addresses to integers, safely handling invalid IPs."""
        try:
            def safe_convert(x):
                try:
                    return int(ipaddress.IPv4Address(x))
                except Exception:
                    print(f"Invalid IP: {x}")
                    return np.nan
            return ip_series.apply(safe_convert)
        except Exception as e:
            print(f"Error converting IPs to integers: {str(e)}")
            return None
    # The following function merges fraud transaction data with country information by mapping each transaction's IP (as integer)
    # to the corresponding country using the IP-to-country mapping DataFrame.
    def merge_with_country_data(self, fraud_df, ip_df):
        """Merge fraud data with country information"""
        try:
            if 'ip_int' not in fraud_df.columns:
                raise KeyError("Column 'ip_int' not found in fraud_df. Please convert IPs to integers first.")
            fraud_df['country'] = fraud_df['ip_int'].apply(
                lambda x: self._find_country(x, ip_df))
            return fraud_df
        except Exception as e:
            print(f"Error merging country data: {str(e)}")
            return None
    # The following function takes a DataFrame of IP addresses and returns a Series of their integer representations.
    def _find_country(self, ip_int, ip_df):
        """Helper function to find country for IP"""
        try:
            if ip_int is None or (isinstance(ip_int, float) and np.isnan(ip_int)):
                return 'Unknown'
            country = ip_df[
                (ip_df['lower_bound_ip_address'] <= ip_int) & 
                (ip_df['upper_bound_ip_address'] >= ip_int)
            ]['country'].values
            return country[0] if len(country) > 0 else 'Unknown'
        except Exception as e:
            print(f"Error finding country for IP: {str(e)}")
            return 'Unknown'
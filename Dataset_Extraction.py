import os
import numpy as np
import pandas as pd

# List CSV files
DATASET_DIRECTORY = "C:/Users/Galym Beketai/Downloads/CICIoT2023/"
csv_files = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
csv_files.sort()


def sample_rows(df, percent_rows):

    labels = df['label'].unique()
    dfs_condensed = []

    # Select rows with chosen label
    for label in labels:
        mask = df['label'] == label
        df_by_label = df[mask]

        # Randomly sample some percentage of rows in current class
        sample = df_by_label.sample(frac=percent_rows)
        dfs_condensed.append(sample)

    # Shuffle all samples
    return pd.concat(dfs_condensed, ignore_index=True).sample(frac=1)


# Map IANA Protocol numbers to strings
iana_map = {
    "0": "HOPOPT", "1": "ICMP", "2": "IGMP", "3": "GGP", "4": "IPv4", "5": "ST",
    "6": "TCP", "7": "CBT", "8": "EGP", "9": "IGP", "10": "BBN-RCC-MON", "11": "NVP-II",
    "12": "PUP", "13": "ARGUS (deprecated)", "14": "EMCON", "15": "XNET", "16": "CHAOS",
    "17": "UDP", "18": "MUX", "19": "DCN-MEAS", "20": "HMP", "21": "PRM", "22": "XNS-IDP",
    "23": "TRUNK-1", "24": "TRUNK-2", "25": "LEAF-1", "26": "LEAF-2", "27": "RDP",
    "28": "IRTP", "29": "ISO-TP4", "30": "NETBLT", "31": "MFE-NSP", "32": "MERIT-INP",
    "33": "DCCP", "34": "3PC", "35": "IDPR", "36": "XTP", "37": "DDP", "38": "IDPR-CMTP",
    "39": "TP++", "40": "IL", "41": "IPv6", "42": "SDRP", "43": "IPv6-Route",
    "44": "IPv6-Frag", "45": "IDRP", "46": "RSVP", "47": "GRE", "48": "DSR", "49": "BNA",
    "50": "ESP", "51": "AH", "52": "I-NLSP", "53": "SWIPE (deprecated)", "54": "NARP",
    "55": "MOBILE", "56": "TLSP", "57": "SKIP", "58": "IPv6-ICMP", "59": "IPv6-NoNxt",
    "60": "IPv6-Opts", "62": "CFTP", "64": "SAT-EXPAK", "65": "KRYPTOLAN", "66": "RVD",
    "67": "IPPC", "69": "SAT-MON", "70": "VISA", "71": "IPCV", "72": "CPNX", "73": "CPHB",
    "74": "WSN", "75": "PVP", "76": "BR-SAT-MON", "77": "SUN-ND", "78": "WB-MON",
    "79": "WB-EXPAK", "80": "ISO-IP", "81": "VMTP", "82": "SECURE-VMTP", "83": "VINES",
    "84": "IPTM", "85": "NSFNET-IGP", "86": "DGP", "87": "TCF", "88": "EIGRP",
    "89": "OSPFIGP", "90": "Sprite-RPC", "91": "LARP", "92": "MTP", "93": "AX.25",
    "94": "IPIP", "95": "MICP (deprecated)", "96": "SCC-SP", "97": "ETHERIP", "98": "ENCAP",
    "100": "GMTP", "101": "IFMP", "102": "PNNI", "103": "PIM", "104": "ARIS", "105": "SCPS",
    "106": "QNX", "107": "A/N", "108": "IPComp", "109": "SNP", "110": "Compaq-Peer",
    "111": "IPX-in-IP", "112": "VRRP", "113": "PGM", "114": "", "115": "L2TP", "116": "DDX",
    "117": "IATP", "118": "STP", "119": "SRP", "120": "UTI", "121": "SMP",
    "122": "SM (deprecated)", "123": "PTP", "124": "ISIS over IPv4", "125": "FIRE",
    "126": "CRTP", "127": "CRUDP", "128": "SSCOPMCE", "129": "IPLT", "130": "SPS",
    "131": "PIPE", "132": "SCTP", "133": "FC", "134": "RSVP-E2E-IGNORE",
    "135": "Mobility Header", "136": "UDPLite", "137": "MPLS-in-IP", "138": "manet",
    "139": "HIP", "140": "Shim6", "141": "WESP", "142": "ROHC", "143": "Ethernet",
    "144": "AGGFRAG", "145": "NSH"
}


def iana_convert(df):
    df["Protocol Type"] = df["Protocol Type"].apply(lambda num: iana_map[str(int(num))])
    return df



dtypes = {
    'flow_duration': np.float32,
    'Header_Length': np.uint32,
    'Protocol Type': str,
    'Duration': np.float32,
    'Rate': np.uint32,
    'Srate': np.uint32,
    'Drate': np.float32,
    'fin_flag_number': np.bool_,
    'syn_flag_number': np.bool_,
    'rst_flag_number': np.bool_,
    'psh_flag_number': np.bool_,
    'ack_flag_number': np.bool_,
    'ece_flag_number': np.bool_,
    'cwr_flag_number': np.bool_,
    'ack_count': np.float16,
    'syn_count': np.float16,
    'fin_count': np.uint16,
    'urg_count': np.uint16,
    'rst_count': np.uint16,
    'HTTP': np.bool_,
    'HTTPS': np.bool_,
    'DNS': np.bool_,
    'Telnet': np.bool_,
    'SMTP': np.bool_,
    'SSH': np.bool_,
    'IRC': np.bool_,
    'TCP': np.bool_,
    'UDP': np.bool_,
    'DHCP': np.bool_,
    'ARP': np.bool_,
    'ICMP': np.bool_,
    'IPv': np.bool_,
    'LLC': np.bool_,
    'Tot sum': np.float32,
    'Min': np.float32,
    'Max': np.float32,
    'AVG': np.float32,
    'Std': np.float32,
    'Tot size': np.float32,
    'IAT': np.float32,
    'Number': np.float32,
    'Magnitue': np.float32,
    'Radius': np.float32,
    'Covariance': np.float32,
    'Variance': np.float32,
    'Weight': np.float32,
    'label': str
}


def convert_dtype(df):
    # Adjust data type
    for col, typ in dtypes.items():
        df[col] = df[col].astype(typ)

        # Format column names to lowercase snake
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Fix spelling error in original dataset
    df['magnitude'] = df['magnitue']
    return df.drop(['magnitue'], axis=1)


# Create cyberattack/no cyberattack label maps
def label_map(df_34):
    df_2 = df_34.copy()

    # Adjust label classes
    df_2['benign'] = df_2['label'] == 'BenignTraffic'
    df_2 = df_2.drop(['label'], axis=1)

    return df_2


def write_helper(df, filename, append=True):

    if append:
        df.to_csv(filename + '_2classes.csv', mode='a', index=False, header=False)
    else:
        df.to_csv(filename + '_2classes.csv', index=False)


def combine_csv(csv_files, percent):

    df = label_map(convert_dtype(iana_convert(sample_rows(
        pd.read_csv(DATASET_DIRECTORY + csv_files[0]), percent_rows=percent
    ))))
    write_helper(df, f'C:/Users/Galym Beketai/Downloads/{percent}percent', append=False)
    del df

    print(f"Appending into {percent} csv")
    for csv in csv_files[1:]:
        print(".", end="")

        # Preprocessing
        df = label_map(convert_dtype(iana_convert(sample_rows(
            pd.read_csv(DATASET_DIRECTORY + csv), percent_rows=percent
        ))))

        # Append to CSV
        write_helper(df, f'C:/Users/Galym Beketai/Downloads/{percent}percent')
        del df


combine_csv(csv_files, percent=0.001)
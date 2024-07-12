import base64
import io
import pandas as pd


def parse_contents(contents, filename):
    """decode csv file

    Args:
        contents ([type]): [description]
        filename (str): path

    Returns:
        [type]: [description]
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('latin1')),
                sep=';' , 
                encoding = "ISO-8859-1"
            )
            return df
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return df
    except Exception as e:
        print(e)
        return None

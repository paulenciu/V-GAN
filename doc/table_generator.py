from doc.EncSpaceTable import EncSpaceTable
from doc.MyopicAttentionTable import MyopicAttentionTable
from doc.PixelSpaceTable import PixelSpaceTable


def generate_pixel_space_tables(metric="auc"):
    table = PixelSpaceTable()
    table.generate_table(metric)

def generate_encoded_space_tables(metric="auc", date="09-04", model="L_16_VIT"):
    table = EncSpaceTable()
    table.generate_table(metric=metric, date=date, model=model)

def generate_attention_table_on_baseline(metric="auc"):
    table = MyopicAttentionTable()
    table.generate_table(metric=metric)
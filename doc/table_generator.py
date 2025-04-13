from doc.PixelSpaceTable import PixelSpaceTable


def generate_pixel_space_tables(metric="auc"):
    table = PixelSpaceTable()
    table.generate_table(metric)
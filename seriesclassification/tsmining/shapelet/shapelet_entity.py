class ShapeletEntity:
    def __init__(self,
                 length=None, start_pos=None,
                 series_id=None, class_label=None,
                 content=None, quality=None):
        self.content = content
        self.length = length
        self.start_pos = start_pos
        self.series_id = series_id
        self.quality = quality
        self.class_label = class_label


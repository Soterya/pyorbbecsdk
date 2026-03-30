from pyorbbecsdk import Pipeline, OBSensorType

pipeline = Pipeline()
profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

for i in range(profile_list.get_count()):
    profile = profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
    print(
        i,
        profile.get_width(),
        profile.get_height(),
        profile.get_format(),
        profile.get_fps(),
    )

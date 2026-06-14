/*视频流*/
function CameraStream(ts, id, t_name) {
    if(isClickSameFunc(id)){
        return;
    }
    loadStreamTemplate(ts);
    postUnitV2Func(id, t_name);
}
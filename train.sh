# 功能说明：
# -gpu 指定gpu
# -c 指定重新训练还是第一次训练
# -n 指定继续训练的第几轮权重文件，查看backup_1文件
CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES
diagram_data="diagram.data"
# log_dir="logdir"
num=900
while getopts ":c:g:n:" opt
do
    case $opt in
        n)
            num=$OPTARG
        ;;
        c)
            case $OPTARG in
            f)
                darknet/darknet detector train model/diagram_1.data model/yolov3-voc.cfg model/darknet53.conv.74 |tee logdir/log/first_train_yolov3.log
            ;;
            r)
                darknet/darknet detector train model/$diagram_data model/yolov3-voc.cfg backup/yolov3-voc_900.weights |tee logdir/log/train_yolov3.log
            ;;
            esac
        ;;
        g)
            CUDA_VISIBLE_DEVICES=$OPTARG
            export CUDA_VISIBLE_DEVICES
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done
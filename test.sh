CUDA_VISIBLE_DEVICES=1
diagram_data="diagram.data"
backup="final_model"
num="11_28"
export CUDA_VISIBLE_DEVICES
test_dir=""
cd darknet
while getopts ":p:d:c:g:n:" opt
do
    case $opt in
        p)
            test_dir=$OPTARG
        ;;
        d)
             backup=$OPTARG
        ;;
        n)
            num=$OPTARG
        ;;
        c)
            case $OPTARG in
            test)
                ./darknet detector test ../model/$diagram_data ../model/yolov3-voc-test.cfg ../$backup/yolov3-voc_$num.weights $test_dir
            ;;
            vaild)
                ./darknet detector valid ../model/$diagram_data ../model/yolov3-voc.cfg ../$backup/yolov3-voc_$num.weights
            ;;
            recall)
                ./darknet detector recall ../model/$diagram_data ../model/yolov3-voc-test.cfg ../$backup/yolov3-voc_$num.weights
            ;;
            ?)
                echo "未知参数"
                exit 1;;
            esac

        ;;
        g)
            CUDA_VISIBLE_DEVICES=$OPTARG
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done
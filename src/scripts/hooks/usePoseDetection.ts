import * as React from 'react';
import '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';

interface IProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  options?: IOptions;
}

interface IOptions {
  imageScaleFactor: number; // ネットワークに画像を渡す際の縮尺。0.2 ～ 1.0を範囲とし、大きいければ大きいほど予測の正確さが増す。ただしその分処理速度が遅くなる。
  flipHorizontal: boolean; // 結果を左右反転する場合にtrue。鏡像がONになっているようなWEBカメラで使う
  outputStride: number; // モデルに渡すストリームのストライド。大きければ大きいほど予測に用いるインプットのモーションが増え、精度が上がる（多分）
}

export interface IEyesPotion {
  left: posenet.Keypoint['position'];
  right: posenet.Keypoint['position'];
}

interface ISize {
  width: number;
  height: number;
}

const DEFALT_OPTIONS = {
  imageScaleFactor: 0.2,
  flipHorizontal: false,
  outputStride: 16,
};

const DEFAULT_EYE_POSITION = {x: 0, y: 0};
const DEFAULT_EYES_POSITION = {left: DEFAULT_EYE_POSITION, right: DEFAULT_EYE_POSITION};

const usePoseDetection = ({videoRef, options = DEFALT_OPTIONS}: IProps) => {
  const [net, setNet] = React.useState<null | posenet.PoseNet>(null);
  const [eyesPotions, setEyesPotions] = React.useState<IEyesPotion>(DEFAULT_EYES_POSITION);
  React.useEffect(() => {
    loadNet();
  }, []);

  React.useEffect(() => {
    if (!(net && videoRef.current)) return;
    detectPoseInRealTime(net, videoRef.current);
  }, [net]);

  const loadNet = React.useCallback(async () => {
    setNet(await posenet.load());
  }, []);

  const detectPoseInRealTime = React.useCallback(
    async (net: posenet.PoseNet, video: HTMLVideoElement) => {
      const pose = await net.estimateSinglePose(video, ...Object.values(options));
      const videoSize = {width: video.videoWidth, height: video.videoHeight};
      const [left, right] = filterEyes(pose.keypoints)
        .map(extractPosition)
        .map(position => getRatioToVideoSize(position, videoSize));
      setEyesPotions({left, right});
      requestAnimationFrame(() => detectPoseInRealTime(net, video));
    },
    []
  );

  const getRatioToVideoSize = React.useCallback(
    (position: posenet.Keypoint['position'], videoSize: ISize) => {
      const {x, y} = position;
      const {width, height} = videoSize;
      return {x: x / width, y: y / height};
    },
    []
  );

  const extractPosition = (keypoint: posenet.Keypoint): posenet.Keypoint['position'] =>
    keypoint.position;

  const filterEyes = (keypoints: posenet.Keypoint[]) =>
    keypoints.filter(keypoint => keypoint.part.match(/Eye/));

  return {eyesPotions};
};
export default usePoseDetection;

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export interface GazePoint {
    x: number; // pixels
    y: number; // pixels
    u: number; // 0..1
    v: number; // 0..1
}

export type CalibrationStep =
    | 'idle'
    | 'collecting_center'
    | 'awaiting_top_center'
    | 'collecting_top_center'
    | 'awaiting_top_right'
    | 'collecting_top_right'
    | 'awaiting_middle_right'
    | 'collecting_middle_right'
    | 'awaiting_bottom_right'
    | 'collecting_bottom_right'
    | 'awaiting_bottom_center'
    | 'collecting_bottom_center'
    | 'awaiting_bottom_left'
    | 'collecting_bottom_left'
    | 'awaiting_middle_left'
    | 'collecting_middle_left'
    | 'awaiting_top_left'
    | 'collecting_top_left'
    | 'done';

export const SCREEN_TARGETS: Record<string, { u: number; v: number; label: string }> = {
    collecting_center: { u: 0.5, v: 0.5, label: 'Center' },
    collecting_top_center: { u: 0.5, v: 0.1, label: 'Top Center' },
    collecting_top_right: { u: 0.9, v: 0.1, label: 'Top Right' },
    collecting_middle_right: { u: 0.9, v: 0.5, label: 'Middle Right' },
    collecting_bottom_right: { u: 0.9, v: 0.9, label: 'Bottom Right' },
    collecting_bottom_center: { u: 0.5, v: 0.9, label: 'Bottom Center' },
    collecting_bottom_left: { u: 0.1, v: 0.9, label: 'Bottom Left' },
    collecting_middle_left: { u: 0.1, v: 0.5, label: 'Middle Left' },
    collecting_top_left: { u: 0.1, v: 0.1, label: 'Top Left' },
};

// Simple Ridge Regression solver
type Matrix = number[][];
type Vector = number[];

function transpose(m: Matrix): Matrix {
    return m[0].map((_, col) => m.map(row => row[col]));
}

function multiply(a: Matrix, b: Matrix): Matrix {
    const rA = a.length, cA = a[0].length, cB = b[0].length;
    const res: Matrix = Array.from({ length: rA }, () => new Array(cB).fill(0));
    for (let r = 0; r < rA; r++) {
        for (let c = 0; c < cB; c++) {
            for (let i = 0; i < cA; i++) res[r][c] += a[r][i] * b[i][c];
        }
    }
    return res;
}

function multiplyVec(m: Matrix, v: Vector): Vector {
    return m.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));
}

function addMatrix(a: Matrix, b: Matrix): Matrix {
    return a.map((row, r) => row.map((val, c) => val + b[r][c]));
}

function identity(size: number): Matrix {
    return Array.from({ length: size }, (_, i) => {
        const row = new Array(size).fill(0);
        row[i] = 1;
        return row;
    });
}

function invert(matrix: Matrix): Matrix | null {
    const n = matrix.length;
    const augmented: Matrix = matrix.map((row, i) => [...row, ...identity(n)[i]]);
    for (let i = 0; i < n; i++) {
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) maxRow = k;
        }
        [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
        const pivot = augmented[i][i];
        if (Math.abs(pivot) < 1e-10) return null;
        for (let j = i; j < 2 * n; j++) augmented[i][j] /= pivot;
        for (let k = 0; k < n; k++) {
            if (k !== i) {
                const factor = augmented[k][i];
                for (let j = i; j < 2 * n; j++) augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }
    return augmented.map(row => row.slice(n));
}

export function ridgeRegression(X: Matrix, y: Vector, lambda = 0.01): Vector | null {
    try {
        const XT = transpose(X);
        const XTX = multiply(XT, X);
        const reg = identity(XTX.length).map(row => row.map(v => v * lambda));
        const inv = invert(addMatrix(XTX, reg));
        if (!inv) return null;
        const term2 = multiply(inv, XT);
        return multiplyVec(term2, y);
    } catch {
        return null;
    }
}

export class GazeTrackerEngine {
    private landmarker: FaceLandmarker | null = null;
    private coeffsU: Vector | null = null;
    private coeffsV: Vector | null = null;

    async init() {
        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
        );
        this.landmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numFaces: 1
        });
    }

    extractFeatures(landmarks: { x: number; y: number; z: number }[]) {
        if (!landmarks || landmarks.length < 478) return null;

        // Eye corner landmarks
        const leftOuter = landmarks[33];
        const leftInner = landmarks[133];
        const rightInner = landmarks[362];
        const rightOuter = landmarks[263];

        // Iris center landmarks
        const leftIris = landmarks[468] || { x: (leftOuter.x + leftInner.x) / 2, y: (leftOuter.y + leftInner.y) / 2 };
        const rightIris = landmarks[473] || { x: (rightOuter.x + rightInner.x) / 2, y: (rightOuter.y + rightInner.y) / 2 };

        const leftCenter = { x: (leftOuter.x + leftInner.x) / 2, y: (leftOuter.y + leftInner.y) / 2 };
        const rightCenter = { x: (rightOuter.x + rightInner.x) / 2, y: (rightOuter.y + rightInner.y) / 2 };

        const leftWidth = Math.hypot(leftInner.x - leftOuter.x, leftInner.y - leftOuter.y) || 0.05;
        const rightWidth = Math.hypot(rightOuter.x - rightInner.x, rightOuter.y - rightInner.y) || 0.05;

        // Eye vector relative to eye center
        const vecLx = (leftIris.x - leftCenter.x) / leftWidth;
        const vecLy = (leftIris.y - leftCenter.y) / leftWidth;
        const vecRx = (rightIris.x - rightCenter.x) / rightWidth;
        const vecRy = (rightIris.y - rightCenter.y) / rightWidth;

        const avgX = (vecLx + vecRx) / 2;
        const avgY = (vecLy + vecRy) / 2;

        return {
            rawX: avgX,
            rawY: avgY,
            features: [1, avgX, avgY]
        };
    }

    trainModel(samples: { features: number[]; target: { u: number; v: number } }[]) {
        if (samples.length < 3) return false;
        const X = samples.map(s => s.features);
        const yU = samples.map(s => s.target.u);
        const yV = samples.map(s => s.target.v);

        const cu = ridgeRegression(X, yU, 0.01);
        const cv = ridgeRegression(X, yV, 0.01);

        if (cu && cv) {
            this.coeffsU = cu;
            this.coeffsV = cv;
            return true;
        }
        return false;
    }

    predict(features: number[], width: number, height: number): GazePoint | null {
        if (this.coeffsU && this.coeffsV) {
            const u = Math.max(0, Math.min(1, this.coeffsU[0] + this.coeffsU[1] * features[1] + this.coeffsU[2] * features[2]));
            const v = Math.max(0, Math.min(1, this.coeffsV[0] + this.coeffsV[1] * features[1] + this.coeffsV[2] * features[2]));
            return { u, v, x: u * width, y: v * height };
        } else {
            // Fallback before calibration: center-offset calculation
            const u = Math.max(0, Math.min(1, 0.5 + features[1] * 3.5));
            const v = Math.max(0, Math.min(1, 0.5 + features[2] * 3.5));
            return { u, v, x: u * width, y: v * height };
        }
    }

    detect(video: HTMLVideoElement, timestamp: number) {
        if (!this.landmarker) return null;
        return this.landmarker.detectForVideo(video, timestamp);
    }
}

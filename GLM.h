#pragma once

// A.1. glm ライブラリーの一部機能は CPP を定義して制御できるようになっています。
#define GLM_FORCE_CUDA
#define GLM_FORCE_CXX14
#define GLM_FORCE_INLNIE
#define GLM_SWIZZLE
// 1. glm ライブラリーの基本機能が含まれます。
#include <glm/glm.hpp>
// 2. glm ライブラリーの拡張機能のうち既に仕様が安定した機能が glm/gtc に含まれます。
// #include <glm/gtc/constants.hpp>
// 3. glm ライブラリーの拡張機能のうち試験的に実装されている機能が glm/gtx に含まれます。
// #include <glm/gtx/color_space.hpp>
// 4. glm ライブラリーの拡張機能をひとまとめに取り込みたい場合には glm/ext.hpp を使います。
#include <glm/ext.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

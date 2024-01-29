import cv2,random
import numpy as np
from PIL import Image,ImageDraw


FRONTIER_COLOR=(0,140,255) #BGR,orange
FRONTIER_CANDIDATEPOINTS_COLOR=(0,255,255)#bgr yellow
class MyFrontierAgent:
    def __init__(self,seed=123,use_contour_sampling=True,show_animation=False):
        self.frontier_target = None
        self.seed = seed
        #self._rng = random.Random(seed)
        self._rng = random.Random()
        self.use_contour_sampling = use_contour_sampling
        self.show_animation = show_animation
        self.frontier_mask = None #will be updated in sample_frontier_target, bool ndarray M*M

    def ans2frontier_format(self,map_states):
        #convert the map obtained from ANS to the occ_map format required by samle_frontier_target()
        #map_states - (bs, 2, M, M) world map with channel 0 representing occupied
        #regions (1s) and channel 1 representing explored regions (1s)
        #return occ_map - occupancy map with the following color coding:
        #(0, 0, 255) is occupied region
        #(255, 255, 255) is unknown region
        #(0, 255, 0) is free region
        b,w,h=map_states.shape[0],map_states.shape[2],map_states.shape[3]
        maps_dict={}
        maps_dict["explored_map"] = (map_states[:, 1] > 0.5).float()  # (bs, M, M)
        maps_dict["occ_space_map"] = (map_states[:, 0] > 0.5).float() * maps_dict["explored_map"]  # (bs, M, M),value 1 is occupancy
        maps_dict["free_space_map"] = (map_states[:, 0] <= 0.5).float() * maps_dict["explored_map"]#value 1 is free space
        #convert to numpy cpu
        for key,value in maps_dict.items():
            maps_dict[key] = value.cpu().data.numpy()

        map_frontier_format = np.zeros([b,3,h,w],dtype=np.uint8)#0~255

        #unknown region
        map_frontier_format[:,0,:,:]+= np.uint8((1- maps_dict["explored_map"])*255)
        map_frontier_format[:, 1, :, :] += np.uint8((1 - maps_dict["explored_map"]) * 255)
        map_frontier_format[:, 2, :, :] += np.uint8((1 - maps_dict["explored_map"]) * 255)
        #free space
        map_frontier_format[:, 1, :, :] +=np.uint8(maps_dict["free_space_map"]*255)
        #map_frontier_format[:, 2, :, :] += np.uint8(maps_dict["free_space_map"] * 255)
        #occupied region
        map_frontier_format[:, 2, :, :] += np.uint8(maps_dict["occ_space_map"] * 255)
        if self.show_animation:
            img_pil = Image.fromarray((map_frontier_format.transpose((0, 2, 3, 1))[0]))#to [h,w,c]
            img_pil.save("images_output/frontier.png")
            #cv2.imwrite("images_output/explored_map.png",maps_dict["explored_map"])
            #cv2.imwrite("images_output/occ_space_map.png", maps_dict["occ_space_map"])
            #cv2.imwrite("images_output/free_space_map.png", maps_dict["free_space_map"])
       #  map_frontier_format = np.transpose(map_frontier_format,(1,0,2,3))#to 3,b,h,w
       #  map_frontier_format =np.reshape(map_frontier_format,(3,-1))#to 3,b*h*w
       #
       #  #(255, 255, 255) is unknown region
       #  unknown_index = np.squeeze(maps_dict["explored_map"].reshape((b, -1)) == 0.0)
       #  map_frontier_format[0,unknown_index]=255
       #  map_frontier_format[1, unknown_index] = 255
       #  map_frontier_format[2, unknown_index] = 255
       #  #(0, 0, 255) is occupied region
       #  occupied_index = np.squeeze(maps_dict["occ_space_map"].reshape((b, -1)) == 1.0)
       #  map_frontier_format[0, occupied_index] = 0
       #  map_frontier_format[1, occupied_index] = 0
       #  map_frontier_format[2, occupied_index] = 255
       # # (0, 255, 0) is free region
       #  free_index = np.squeeze(maps_dict["free_space_map"].reshape((b, -1)) == 1.0)
       #  map_frontier_format[0, free_index] = 0
       #  map_frontier_format[1, free_index] = 255
       #  map_frontier_format[2, free_index] = 255
       #
       #  map_frontier_format =  map_frontier_format.reshape((3,b,h,w))
       #  map_frontier_format = np.transpose(map_frontier_format,(1,0,2,3))
        return map_frontier_format#[b,3,h,w]
    def batch_sample_frontier_target(self,occ_map_batch):
        #occ_map_batch [b,3,M,M]
        b,h,w = occ_map_batch.shape[0],occ_map_batch.shape[2],occ_map_batch.shape[3]
        frontier_target_batch = np.zeros((b,2))
        frontier_mask_batch = np.zeros((b,h,w))
        for batchi in range(b):
            occ_map = occ_map_batch[batchi]
            frontier_target,frontier_mask = self.sample_frontier_target(occ_map)
            frontier_target_batch[batchi,:] = frontier_target
            frontier_mask_batch[batchi,:] = frontier_mask
        return frontier_target_batch,frontier_mask_batch
    def sample_frontier_target(self, occ_map):
        """
        Inputs:
            occ_map - occupancy map with the following color coding:
                      (0, 0, 255) is occupied region
                      (255, 255, 255) is unknown region
                      (0, 255, 0) is free region
        """
        #self.occ_buffer.fill(0)
        self._time_elapsed_for_target = 0
        self._failure_count = 0
        self. map_size = occ_map.shape[2]

        occ_map = occ_map.transpose((1,2,0))#[c,h,w] to [h,w,c]

        unknown_mask = np.all(occ_map == (255, 255, 255), axis=-1).astype(np.uint8)
        free_mask = np.all(occ_map == (0, 255, 0), axis=-1).astype(np.uint8)

        unknown_mask_shiftup = np.pad(
            unknown_mask, ((0, 1), (0, 0)), mode="constant", constant_values=0
        )[1:, :]
        unknown_mask_shiftdown = np.pad(
            unknown_mask, ((1, 0), (0, 0)), mode="constant", constant_values=0
        )[:-1, :]
        unknown_mask_shiftleft = np.pad(
            unknown_mask, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )[:, 1:]
        unknown_mask_shiftright = np.pad(
            unknown_mask, ((0, 0), (1, 0)), mode="constant", constant_values=0
        )[:, :-1]

        frontier_mask = (
                                (free_mask == unknown_mask_shiftup)
                                | (free_mask == unknown_mask_shiftdown)
                                | (free_mask == unknown_mask_shiftleft)
                                | (free_mask == unknown_mask_shiftright)
                        ) & (free_mask == 1)
        #self.frontier_mask =  frontier_mask
        frontier_idxes = list(zip(*np.where(frontier_mask)))
        if len(frontier_idxes) > 0:
            if self.use_contour_sampling:
                frontier_img = frontier_mask.astype(np.uint8) * 255
                # # Reduce size for efficiency; converted back later
                # scaling_factor = frontier_mask.shape[0] / 200.0
                # frontier_img = cv2.resize(
                #     frontier_img,
                #     None,
                #     fx=1.0 / scaling_factor,
                #     fy=1.0 / scaling_factor,
                #     interpolation=cv2.INTER_NEAREST,
                # )
                # Add a single channel
                frontier_img = frontier_img[:, :, np.newaxis]
                contours, _ = cv2.findContours(frontier_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_orin=contours.copy()
                if len(contours) == 0:
                    tgt = self._rng.choice(frontier_idxes)  # (y, x)
                else:
                    contours_length = [len(contour) for contour in contours]
                    contours = list(zip(contours, contours_length))
                    sorted_contours = sorted(contours, key=lambda x: x[1], reverse=True)

                    #contours = sorted_contours[:3]
                    contours = sorted_contours[:3] if len(sorted_contours) >= 3 else sorted_contours
                    # Randomly pick one of the longest contours
                    # To introduce some stochasticity in case the agent is stuck
                    max_contour = self._rng.choice(contours)[0]
                    #print random choice
                    # aa = self._rng.choice(contours)[0]
                    # for tmpi, tmpitem in enumerate(contours):
                    #     if np.all(aa == tmpitem[0]):
                    #         print("random selected contour index:", tmpi)

                    # Pick a random sample from the longest contour
                    tgt = self._rng.choice(max_contour)[0]  # Each point is [[x, y]] for some reason
                    # # Scale it back to original image size
                    # # Convert it to (y, x) convention as this will be reversed next
                    # tgt = (int(tgt[1] * scaling_factor), int(tgt[0] * scaling_factor))
            else:
                tgt = self._rng.choice(frontier_idxes)  # (y, x)

            self.frontier_target = (
                np.clip(tgt[1], 1, self.map_size - 2).item(),
                np.clip(tgt[0], 1, self.map_size - 2).item(),
            )  # (x, y) (row col)
            #draw contours and contours candidates; contours are orange, 3 candidates are in yellow
            frontier_img_draw = np.concatenate([frontier_img, frontier_img, frontier_img], axis=-1)
            frontier_img_draw = cv2.drawContours(frontier_img_draw, contours_orin, -1, FRONTIER_COLOR, thickness=1)
            contours_cand = [coni[0] for coni in contours]
            frontier_img_draw = cv2.drawContours(frontier_img_draw, contours_cand, -1, FRONTIER_CANDIDATEPOINTS_COLOR, thickness=1)

        else:
            self.frontier_target = (self.map_size // 2 + 4, self.map_size // 2 + 4)
            frontier_img_draw = np.zeros_like(frontier_mask)
        self.frontier_img_draw = frontier_img_draw
        self.free_mask = free_mask
        frontier_img_draw = cv2.circle(frontier_img_draw.astype('uint8'), (self.frontier_target[1],self.frontier_target[0]), 6, (0, 0, 255), -1, )#red is for long term goal,(col,row)

        if self.show_animation:
            occ_map_copy = np.copy(occ_map[...,::-1])#cv2 [RGB] TO [B,G,R], inorder to be consistent with PIL image
            occ_map_copy = cv2.circle(occ_map_copy, (self.frontier_target[1],self.frontier_target[0]), 3, (0, 0, 255), -1)# free space green; occupied blue; target point red
            frontier_img_copy = np.copy(frontier_img[...,::-1])
            frontier_img_copy = cv2.circle(frontier_img_copy, (self.frontier_target[1],self.frontier_target[0]), 3, (0, 0, 255), -1)
            cv2.imwrite("images_output/occ_map_targetpoint.png",occ_map_copy)
            cv2.imwrite("images_output/frontier_map_targetpoint.png", frontier_img_copy)
            cv2.imwrite("images_output/frontier_countours.png", frontier_img_draw)
            cv2.imwrite("images_output/frontier_mask.png",frontier_mask.astype(np.uint8) * 255)
        #     cv2.imshow("Occupancy map with target", np.flip(occ_map_copy, axis=2))
        #     cv2.imshow("Frontier mask", frontier_mask.astype(np.uint8) * 255)
        #     cv2.waitKey(10)
        self.frontier_target = (self.frontier_target[1],self.frontier_target[0])#(row,col) to (col,row)
        return self.frontier_target,frontier_mask


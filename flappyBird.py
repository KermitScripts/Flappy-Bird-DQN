import pygame
import random
import time
import math
from pygame.locals import *
import pygame.draw
from DQN.dqn import *
from DQN.helper import *

# VARIABLES
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 500

PIPE_GAP = 150

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                       pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                       pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]
        self.speed = SPEED
        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.passed = False
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED

class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED

class Score(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.digit_images = [pygame.image.load('assets/sprites/' + str(i) + '.png') for i in range(10)]
        self.score = 0
        self.image = pygame.Surface((100, 40), pygame.SRCALPHA) 
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 2, 30) 

        # Clear the score surface with a transparent background only once
        self.image.fill((0, 0, 0, 0))

    def update(self):
        # Update the score image based on the current score
        self.score += 1
        score_str = str(self.score)
        x_position = (self.image.get_width() - len(score_str) * self.digit_images[0].get_width()) // 2

        # Clear the score surface with a transparent background
        self.image.fill((0, 0, 0, 0))

        for digit_char in score_str:
            if digit_char.isdigit():
                digit_image = self.digit_images[int(digit_char)]
                self.image.blit(digit_image, (x_position, 0))
                x_position += digit_image.get_width()

class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')
        self.background = pygame.transform.scale(pygame.image.load('assets/sprites/background-day.png').convert_alpha(), (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.bird_group = pygame.sprite.Group()
        self.ground_group = pygame.sprite.Group()
        self.pipe_group = pygame.sprite.Group()
        self.score_group = pygame.sprite.Group()
        self.clock = pygame.time.Clock()
        self.score = 0
        self.restart()

    def restart(self):
        self.bird_group.empty()
        self.ground_group.empty()
        self.pipe_group.empty()
        self.score_group.empty()

        bird = Bird()
        self.bird_group.add(bird)

        for i in range(2):
            ground = Ground(GROUND_WIDTH * i)
            self.ground_group.add(ground)

        for i in range(2):
            pipes = self.get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.add(pipes[0]) # bottom pipe? 
            self.pipe_group.add(pipes[1]) # top pipe?

        score_sprite = Score()
        self.score_group.add(score_sprite)

        self.score = 0

    def distance_to_pipe(self, bird, pipe):

        bird_center_x = bird.rect.centerx
        bird_center_y = bird.rect.centery
        pipe_center_x = (pipe.rect.left + pipe.rect.right) / 2
        pipe_bottom_y = pipe.rect.top
        pipe_top_y = pipe_bottom_y - PIPE_GAP 

        distance_to_pipe = pipe.rect.left - bird.rect.right
        distance_to_top_pipe = bird_center_y - pipe_top_y
        distance_to_bottom_pipe = pipe_bottom_y - bird_center_y

        # Draw lines to visualize the distances for debug purposes
        #pygame.draw.line(self.screen, (255, 0, 0), (bird_center_x, bird_center_y), (bird_center_x, pipe_top_y), 2) 
        #pygame.draw.line(self.screen, (0, 255, 0), (bird_center_x, bird_center_y), (bird_center_x, pipe_bottom_y), 2)
        #pygame.display.update()

        return distance_to_pipe, distance_to_top_pipe, distance_to_bottom_pipe

    def learn(self):
        all_rewards = []
        episode_rewards = []
        timesteps =  10000
        dqn = DQN()   

        if os.path.exists('dqn_model.pth'):
            dqn.load_model('dqn_model.pth')
        
        for timestep in range(1, timesteps + 1):
            
            sys.stdout.write('\rTimestep: {}/{} '.format(timestep, timesteps))
            sys.stdout.flush()

            self.restart()
            bird = self.bird_group.sprites()[0]
            bird_y_velocity = bird.speed           
            
            pipe = self.find_closest_pipe(bird)
            distance_to_pipe, distance_to_top_pipe, distance_to_bottom_pipe = self.distance_to_pipe(bird, pipe)       
            
            obs = [bird_y_velocity, distance_to_pipe, distance_to_top_pipe, distance_to_bottom_pipe]
            reward = 0
            terminated = False
            counter = 0
            while terminated == False:

                epsilon = dqn.epsilon_by_timestep(timestep)
                action = dqn.predict(obs,epsilon)
                
                self.clock.tick(50)
    
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                if action == 1:
                    self.bird_group.sprites()[0].bump()
                        
                self.screen.blit(self.background, (0, 0))
    
                if self.is_off_screen(self.ground_group.sprites()[0]):
                    self.ground_group.remove(self.ground_group.sprites()[0])
                    new_ground = Ground(GROUND_WIDTH - 20)
                    self.ground_group.add(new_ground)
    
                if self.is_off_screen(self.pipe_group.sprites()[0]):
                    self.pipe_group.remove(self.pipe_group.sprites()[0])
                    self.pipe_group.remove(self.pipe_group.sprites()[0])
                    pipes = self.get_random_pipes(SCREEN_WIDTH * 2)
                    self.pipe_group.add(pipes[0])
                    self.pipe_group.add(pipes[1])
    
                self.bird_group.update()
                self.ground_group.update()
                self.pipe_group.update()

                self.score_group.draw(self.screen)
                self.bird_group.draw(self.screen)
                self.pipe_group.draw(self.screen)
                self.ground_group.draw(self.screen)
                
                pygame.display.update()

                bird = self.bird_group.sprites()[0]
                bird_y_velocity = bird.speed
                bird_x_position = bird.rect[0]
        
                pipe = self.find_closest_pipe(bird)
                distance_to_pipe, distance_to_top_pipe, distance_to_bottom_pipe = self.distance_to_pipe(bird, pipe)  

                next_obs = [bird_y_velocity, distance_to_pipe, distance_to_top_pipe, distance_to_bottom_pipe]

                if self.check_collision(bird):
                    reward += -10
                    time.sleep(1)
                    terminated = True
                elif bird_x_position  > pipe.rect.left and pipe.passed == False:
                    pipe.passed = True
                    reward += 10  # Give a reward of +10 when the bird passes a pipe
                    self.score_group.update()
                elif counter % 5 == 0 :#consider making the reward diffirences larger
                    reward += (1 / ((distance_to_top_pipe + distance_to_bottom_pipe) ** 0.5))

                dqn.replay_buffer.put(obs, action, reward, next_obs, terminated)
                obs = next_obs
                episode_rewards.append(reward)

                counter += 1

            if terminated or len(episode_rewards) >= 500:
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []

            if len(dqn.replay_buffer) > dqn.batch_size:
                loss = dqn.compute_msbe_loss()
                dqn.optim_dqn.zero_grad()
                loss.backward()
                dqn.optim_dqn.step()

            #Sync the target network
            if timestep % dqn.sync_after == 0:
                dqn.dqn_target_net.load_state_dict(dqn.dqn_net.state_dict())

            #if timestep % 1000 == 0:
            #    episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
            
            if timestep % 50 == 0:
                dqn.save_model('dqn_model.pth')

    def check_collision(self, bird):
        return (
            pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask)
            or pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask)
            or bird.rect.top < 0
        )

            
    def find_closest_pipe(self, bird):
        closest_pipe = None
        distance = float('inf')

        for pipe in self.pipe_group:
            if pipe.rect.right < bird.rect.left:
                # Bird has passed this pipe, continue to the next pipe
                continue

            distance_to_pipe = pipe.rect.left - bird.rect.right

            if distance_to_pipe < distance:
                distance = distance_to_pipe
                closest_pipe = pipe

        return closest_pipe

    def is_off_screen(self, sprite):
        return sprite.rect[0] < -(sprite.rect[2])

    def get_random_pipes(self, xpos):
        size = random.randint(100, 300)
        pipe = Pipe(False, xpos, size)
        pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
        return pipe, pipe_inverted


if __name__ == "__main__":
    game = FlappyBirdGame()
    game.learn()

import types
from os import path
from mayavi import mlab
from tvtk.api import tvtk
from pyface.timer.api import Timer
from traits.api import HasTraits, Button, Instance, Range, Bool, Int, Directory, String, Tuple, Event, on_trait_change
from traitsui.api import View, Group, Item, RangeEditor, HGroup, Tabbed, TupleEditor
from pyface.api import ProgressDialog


__all__ = [
    "Animator", 
    "animate"
]


class Animator(HasTraits):
    start = Button('Start Animation')
    stop = Button('Stop Animation')
    next_frame = Button('+')
    prev_frame = Button('-')
    delay = Range(10, 100000, 500)
    loop = Bool(True)

    current_frame = Int(-1)
    _last_frame = Int()

    # TODO use Range(high="trait name")
    render_from_frame = Int()
    render_to_frame = Int()
    render_animation = Button()
    is_rendering = Bool(False) # indicator bool is True when rendering
    is_rendering_animation = Bool(False)

    render_directory = Directory("/tmp", exists=False)
    render_name_pattern = String("frame_%05d.png")

    magnification = Range(1, 128)
    fix_image_size = Bool(False)
    image_size = Tuple(Int(1280), Int(720))

    render = Event()

    enable_cameraman = Bool(False)
    set_keyframe = Button()
    remove_keyframe = Button()

    timer = Instance(Timer)

    traits_view = View( Tabbed(
            Group(
                HGroup(
                    Item('start', show_label=False),
                    Item('stop', show_label=False),
                    Item('next_frame', show_label=False, enabled_when='current_frame < last_frame'),
                    Item('prev_frame', show_label=False, enabled_when='current_frame > 1'),
                ),
                HGroup(
                    Item(name = 'loop'),
                    Item(name = 'delay'),
                ),
                Item(name = 'current_frame', 
                     editor=RangeEditor(is_float=False, high_name='_last_frame', mode='slider')),
                Group(
                    HGroup(
                        Item(name = 'enable_cameraman', label='enabled'),
                        Item(name = 'set_keyframe', show_label=False),
                        Item(name = 'remove_keyframe', show_label=False),
                        Item(name = 'interpolation_type', object='object._camera_interpolator'),
                    ),
                label = 'Cameraman',
                ),
            label = 'Timeline',
            ),
            Group(
                HGroup(
                    Item('fix_image_size', label="Set Image Size"),
                    Item('magnification', visible_when='not fix_image_size', label='Magnification'),
                    Item('image_size', visible_when='fix_image_size', show_label=False, editor=TupleEditor(cols=2, labels=['W', 'H'])),
                ),
                Item("_"),
                Item("render_directory", label="Target Dir"),
                Item("render_name_pattern", label="Filename Pattern"),
                Item("_"),
                HGroup(
                    Item("render_from_frame", label="from",
                         editor=RangeEditor(is_float=False, low=0, high_name='render_to_frame')),
                    Item("render_to_frame", label="to",
                         editor=RangeEditor(is_float=False, low_name='render_from_frame', high_name='_last_frame')),
                ),
                Item("render_animation", show_label=False),
                label = "Render",
            ),
        ),
        title = 'Animation Controller', 
        buttons = [])


    def __init__(self, num_frames, callable, millisec=40, figure=None, play=True, *args, **kwargs):
        HasTraits.__init__(self)
        self.delay = millisec
        self._last_frame = num_frames - 1
        self._callable = callable
        if figure is None:
            figure = mlab.gcf()
        self._figure = figure
        self._camera_interpolator = tvtk.CameraInterpolator(interpolation_type='spline')
        self._t_keyframes = {}
        self.render_to_frame = self._last_frame
        self.timer = Timer(millisec, self._on_timer, *args, **kwargs)
        if not play:
            self.stop = True
        self._internal_generator = None
        self.current_frame = 0
        self.on_trait_change(self._render, "render, current_frame", dispatch="ui")

    def _render(self):
        self.is_rendering = True
        if self._internal_generator is not None:
            try:
                self._internal_generator.next()
            except StopIteration: # is ok since generator should yield just once to render
                pass 
            except: # catch and re-raise other errors
                raise
            else:
                raise "The render function should be either a simple function or a generator that yields just once to render"
        # before we call the user function, we want to disallow rendering
        # this speeds up animations that use mlab functions
        scene = self._figure.scene
        scene.disable_render = True
        r = self._callable(self.current_frame)
        if isinstance(r, types.GeneratorType):
            r.next()
            # save away generator to yield when another frame has to be displayed
            self._internal_generator = r
        # render scene without dumb hourglass cursor, 
        # can be prevented by setting _interacting before calling render
        old_interacting = scene._interacting
        if self._camera_interpolator.number_of_cameras >= 2 and self.enable_cameraman:
            t = self.current_frame / float(self._last_frame)
            self._camera_interpolator.interpolate_camera(t, mlab.get_engine().current_scene.scene.camera)
            mlab.gcf().scene.renderer.reset_camera_clipping_range()
        scene._interacting = True
        scene.disable_render = False
        scene.render()
        scene._interacting = old_interacting
        self.is_rendering = False

    @on_trait_change('set_keyframe')
    def _set_keyframe(self):
        t = self.current_frame / float(self._last_frame)
        self._camera_interpolator.add_camera(t, mlab.get_engine().current_scene.scene.camera)
        self._t_keyframes[self.current_frame] = t

    def _next_frame_fired(self):
        self.current_frame += 1

    def _prev_frame_fired(self):
        self.current_frame -= 1

    @on_trait_change('remove_keyframe')
    def _remove_keyframe(self):
        if self.current_frame in self._t_keyframes:
            self._camera_interpolator.remove_last_keyframe(self._t_keyframes[self.current_frame])

    def _on_timer(self, *args, **kwargs):
        if self.loop or self.current_frame != self._last_frame:
            self.current_frame = (self.current_frame + 1) % (self._last_frame + 1)
        else:
            self.stop = True

    def _delay_changed(self, value):
        t = self.timer
        if t is None:
            return
        if t.IsRunning():
            t.Stop()
            t.Start(value)

    def _start_fired(self):
        if not self.loop and self.current_frame == self._last_frame:
            self.current_frame = 0
        self.timer.Start(self.delay) 

    def _stop_fired(self):
        self.timer.Stop()

    def _render_animation_fired(self):
        self.stop = True
        n_frames_render = self.render_to_frame - self.render_from_frame
        # prepare the render window
        renwin = self._figure.scene.render_window
        aa_frames = renwin.aa_frames
        renwin.aa_frames = 8
        renwin.alpha_bit_planes = 1
        # turn on off screen rendering
        renwin.off_screen_rendering = True
        # set size of window
        if self.fix_image_size:
            orig_size = renwin.size
            renwin.size = self.image_size
        # render the frames
        progress = ProgressDialog(title="Rendering", max=n_frames_render, 
                                  show_time=True, can_cancel=True)
        progress.open()
        self.is_rendering_animation = True
        for frame in xrange(self.render_from_frame, self.render_to_frame + 1):
            # move animation to desired frame, this will also render the scene
            self.current_frame = frame
            # prepare window to image writer
            render = tvtk.WindowToImageFilter(input=renwin, magnification=1)#, input_buffer_type='rgba')
            if not self.fix_image_size:
                render.magnification = self.magnification
            exporter = tvtk.PNGWriter(input=render.output,
                          file_name=path.join(self.render_directory, self.render_name_pattern % frame))
            exporter.write()
            do_continue, skip = progress.update(frame - self.render_from_frame)
            if not do_continue:
                break
        # reset the render window to old values
        renwin.aa_frames = aa_frames
        if self.fix_image_size:
            renwin.size = orig_size
        renwin.off_screen_rendering = False
        self.is_rendering_animation = False
        progress.close()


def animate(num_frames, delay=40, ui=True, fig=None, play=True):
    class Wrapper(object):
        # The wrapper which calls the decorated function.
        def __init__(self, function):
            self.func = function

        def __call__(self, *args, **kw):
            a = Animator(num_frames, self.func, delay, fig, play)
            if ui:
                a.edit_traits()
            return a

    def _wrapper1(function):
        # Needed to create the Wrapper in the right scope.
        w = Wrapper(function)
        return w

    return _wrapper1

